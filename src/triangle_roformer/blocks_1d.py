from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding


@dataclass
class TriangleRoFormerState:
    """Recurrent state for O(1)-memory causal Triangle-RoPE decoding."""

    s_r: torch.Tensor  # [B, H, head_dim // 2, head_dim] (fp32)
    s_i: torch.Tensor  # [B, H, head_dim // 2, head_dim] (fp32)
    step: int = 0
    valid_count: torch.Tensor | None = None  # [B, 1, 1, 1] (fp32) when masked


class TriangleRoFormerBlock1D(nn.Module):
    """Multi-head 1D Triangle Multiplicative Update block with Relative-RoPE.

    Operates on `[B, T, D]` sequences for language modeling, 1D audio, speech,
    or time-series tasks. Supports both bidirectional (`causal=False`) and
    autoregressive (`causal=True`) modes in O(T * H * head_dim^2) time:

    - Non-causal (`causal=False`):
        K_r = (1/T) * sum_{t'} b_r[t']^T c[t']
        K_i = (1/T) * sum_{t'} b_i[t']^T c[t']
        O[t] = (1/sqrt(d_c)) * (a_r[t] K_r + a_i[t] K_i)
      with phase-preserving joint RMS normalization on (K_r, K_i).

    - Causal (`causal=True`):
      Computes the exact prefix contraction K_r[t], K_i[t] over t' <= t in
      memory-bounded chunks (`chunk_size`) and supports optional incremental
      `TriangleRoFormerState` caching for O(1) per-token generation.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        head_dim: int | None = None,
        causal: bool = False,
        ffn_mult: float = 3.0,
        rope_base: float = 10_000.0,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        chunk_size: int = 64,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")
        if heads <= 0:
            raise ValueError(f"heads must be positive, got {heads}.")
        if head_dim is None:
            if dim % heads != 0:
                raise ValueError(
                    f"dim ({dim}) must be divisible by heads ({heads}) when head_dim is None."
                )
            head_dim = dim // heads
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be a positive even integer for RoPE."
            )
        if ffn_mult <= 0.0:
            raise ValueError(f"ffn_mult must be positive, got {ffn_mult}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.causal = causal
        self.chunk_size = chunk_size
        self.dropout = dropout
        ffn_hidden = max(64, int(round(dim * ffn_mult / 64.0)) * 64)

        self.tri_norm = nn.RMSNorm(dim)
        self.in_proj = nn.Linear(dim, self.inner_dim * 6 + dim, bias=False)
        self.a_norm = nn.RMSNorm(self.inner_dim)
        self.b_norm = nn.RMSNorm(self.inner_dim)
        self.c_norm = nn.RMSNorm(self.inner_dim)
        self.rope = RotaryEmbedding(head_dim, base=rope_base)
        self.tri_out_norm = nn.RMSNorm(self.inner_dim)
        self.tri_out = nn.Linear(self.inner_dim, dim, bias=False)

        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn_in = nn.Linear(dim, ffn_hidden * 2, bias=False)
        self.ffn_out = nn.Linear(ffn_hidden, dim, bias=False)

        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self._init_weights(residual_scale)

    def _init_weights(self, residual_scale: float) -> None:
        in_std = 1.0 / math.sqrt(self.dim)
        for proj in (self.in_proj, self.ffn_in):
            nn.init.normal_(proj.weight, std=in_std)
        nn.init.normal_(
            self.tri_out.weight,
            std=(1.0 / math.sqrt(self.inner_dim)) * residual_scale,
        )
        ffn_out_std = (1.0 / math.sqrt(self.ffn_out.in_features)) * residual_scale
        nn.init.normal_(self.ffn_out.weight, std=ffn_out_std)

    def _non_causal_contract(
        self,
        a_r: torch.Tensor,
        a_i: torch.Tensor,
        b_r: torch.Tensor,
        b_i: torch.Tensor,
        c: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, _heads, seq_len, d_c = a_r.shape
        inv_sqrt_dc = 1.0 / math.sqrt(d_c)

        if attention_mask is not None:
            denom = (
                attention_mask.float()
                .sum(dim=-1)
                .clamp_min(1.0)
                .view(batch, 1, 1, 1)
                .to(dtype=b_r.dtype)
            )
            k_r = torch.matmul(b_r.transpose(-2, -1), c) / denom
            k_i = torch.matmul(b_i.transpose(-2, -1), c) / denom
        else:
            inv_t = 1.0 / seq_len
            k_r = torch.matmul(b_r.transpose(-2, -1), c) * inv_t
            k_i = torch.matmul(b_i.transpose(-2, -1), c) * inv_t

        k_rms = (
            (k_r.float().square() + k_i.float().square())
            .mean(dim=-2, keepdim=True)
            .add(1e-6)
            .sqrt()
            .to(dtype=k_r.dtype)
        )
        k_r = k_r / k_rms
        k_i = k_i / k_rms
        return (torch.matmul(a_r, k_r) + torch.matmul(a_i, k_i)) * inv_sqrt_dc

    def _causal_contract(
        self,
        a_r: torch.Tensor,
        a_i: torch.Tensor,
        b_r: torch.Tensor,
        b_i: torch.Tensor,
        c: torch.Tensor,
        attention_mask: torch.Tensor | None,
        state: TriangleRoFormerState | None,
    ) -> tuple[torch.Tensor, TriangleRoFormerState]:
        batch, heads, seq_len, d_c = a_r.shape
        head_dim = c.shape[-1]
        inv_sqrt_dc = 1.0 / math.sqrt(d_c)

        if state is not None:
            s_r = state.s_r.to(device=a_r.device, dtype=torch.float32)
            s_i = state.s_i.to(device=a_r.device, dtype=torch.float32)
            step = state.step
            valid_count = (
                state.valid_count.to(device=a_r.device, dtype=torch.float32)
                if state.valid_count is not None
                else None
            )
        else:
            s_r = torch.zeros(
                batch, heads, d_c, head_dim, device=a_r.device, dtype=torch.float32
            )
            s_i = torch.zeros_like(s_r)
            step = 0
            valid_count = None

        if attention_mask is not None and valid_count is None:
            valid_count = torch.full(
                (batch, 1, 1, 1), float(step), device=a_r.device, dtype=torch.float32
            )

        outputs: list[torch.Tensor] = []
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            chunk_len = end - start

            ar_c = a_r[:, :, start:end, :].float()
            ai_c = a_i[:, :, start:end, :].float()
            br_c = b_r[:, :, start:end, :].float()
            bi_c = b_i[:, :, start:end, :].float()
            c_c = c[:, :, start:end, :].float()

            delta_r = br_c.unsqueeze(-1) * c_c.unsqueeze(-2)
            delta_i = bi_c.unsqueeze(-1) * c_c.unsqueeze(-2)

            cum_r = s_r.unsqueeze(2) + torch.cumsum(delta_r, dim=2)
            cum_i = s_i.unsqueeze(2) + torch.cumsum(delta_i, dim=2)
            s_r = cum_r[:, :, -1]
            s_i = cum_i[:, :, -1]

            if attention_mask is not None:
                assert valid_count is not None
                mask_c = attention_mask[:, start:end].float()
                cum_count = valid_count.view(batch, 1) + torch.cumsum(mask_c, dim=1)
                valid_count = cum_count[:, -1].view(batch, 1, 1, 1)
                denom = cum_count.clamp_min(1.0).view(batch, 1, chunk_len, 1, 1)
            else:
                denom = torch.arange(
                    step + start + 1,
                    step + end + 1,
                    device=a_r.device,
                    dtype=torch.float32,
                ).view(1, 1, chunk_len, 1, 1)

            kr_c = cum_r / denom
            ki_c = cum_i / denom
            k_rms = (
                (kr_c.square() + ki_c.square())
                .mean(dim=-2, keepdim=True)
                .add(1e-6)
                .sqrt()
            )
            kr_c = kr_c / k_rms
            ki_c = ki_c / k_rms

            o_c = (
                torch.matmul(ar_c.unsqueeze(-2), kr_c).squeeze(-2)
                + torch.matmul(ai_c.unsqueeze(-2), ki_c).squeeze(-2)
            ) * inv_sqrt_dc
            outputs.append(o_c.to(dtype=c.dtype))

        out = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=2)
        new_state = TriangleRoFormerState(
            s_r=s_r,
            s_i=s_i,
            step=step + seq_len,
            valid_count=valid_count,
        )
        return out, new_state

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        state: TriangleRoFormerState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, TriangleRoFormerState | None]:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor [B, T, D], got shape {tuple(x.shape)}.")
        batch, seq_len, _dim = x.shape

        if state is not None and not self.causal:
            raise ValueError("Recurrent state caching requires causal=True.")

        h = self.tri_norm(x)
        abc_proj, out_gate = self.in_proj(h).split(
            (self.inner_dim * 6, self.dim), dim=-1
        )
        a_val, a_gate, b_val, b_gate, c_val, c_gate = abc_proj.chunk(6, dim=-1)
        a = self.a_norm(a_val) * torch.tanh(a_gate)
        b = self.b_norm(b_val) * torch.tanh(b_gate)
        c = self.c_norm(c_val) * torch.sigmoid(c_gate)

        a = a.reshape(batch, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        b = b.reshape(batch, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        c = c.reshape(batch, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)

        if attention_mask is not None:
            if attention_mask.shape != (batch, seq_len):
                raise ValueError(
                    f"attention_mask must have shape {(batch, seq_len)}, "
                    f"got {tuple(attention_mask.shape)}."
                )
            mask = attention_mask.to(device=x.device, dtype=b.dtype)[:, None, :, None]
            b = b * mask
            c = c * mask

        offset = state.step if state is not None else 0
        a_r, a_i, b_r, b_i = self.rope.apply_1d(a, b, offset=offset)

        new_state: TriangleRoFormerState | None = None
        if self.causal:
            o, new_state = self._causal_contract(
                a_r, a_i, b_r, b_i, c, attention_mask=attention_mask, state=state
            )
        else:
            o = self._non_causal_contract(
                a_r, a_i, b_r, b_i, c, attention_mask=attention_mask
            )

        tri = o.permute(0, 2, 1, 3).reshape(batch, seq_len, self.inner_dim)
        tri = self.tri_out(self.tri_out_norm(tri)) * torch.sigmoid(out_gate)
        x = x + self.drop(tri)

        h_ffn = self.ffn_norm(x)
        ffn_gate, ffn_val = self.ffn_in(h_ffn).chunk(2, dim=-1)
        ffn = F.silu(ffn_gate) * ffn_val
        out = x + self.drop(self.ffn_out(self.drop(ffn)))

        if return_state:
            return out, new_state
        return out
