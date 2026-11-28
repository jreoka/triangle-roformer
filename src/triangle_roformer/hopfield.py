from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_BETA_MIN = 0.5
_BETA_MAX = 32.0


def _resolve_num_heads(dim: int, num_heads: int | None, target_head_dim: int = 32) -> int:
    if num_heads is not None:
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        return num_heads
    for hd in (target_head_dim, 64, 16, 8, 4, 2):
        if dim >= hd and dim % hd == 0:
            return dim // hd
    return 1


class HopfieldFFN(nn.Module):
    """Modern Continuous Hopfield Network (MCHN) FFN for audio / 2D grids.

    Key design choices over a naive tied-key MCHN:
      - Unit-sphere fixed-point retrieval with `beta` initialized at
        `sqrt(head_dim)` (clamped to `[0.5, 32.0]`) so cosine-similarity logits
        start with well-scaled variance instead of collapsing to uniform weights.
      - Untied `values` table read out on the final step so the FFN can produce
        hetero-associative residual updates rather than projections onto keys.
      - Optional null/sink pattern (`attn_sink=True`) with zero value so silent
        or unremarkable tokens can attend to "nothing".
      - Input-driven multiplicative output gate (`sigmoid(out_gate)`) so radial
        energy/amplitude discarded by L2 normalization is preserved.

    Accepts any `(*leading, dim)` tensor (e.g. `[B, T, F, D]`) and returns a
    residual update of the same shape.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int | None = None,
        num_patterns: int = 256,
        num_steps: int = 2,
        step_size: float = 1.0,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        attn_sink: bool = True,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")
        if num_patterns <= 0:
            raise ValueError(f"num_patterns must be positive, got {num_patterns}.")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")
        if not 0.0 < step_size <= 2.0:
            raise ValueError(f"step_size must be in (0, 2], got {step_size}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        resolved_heads = _resolve_num_heads(dim, num_heads, target_head_dim=32)
        self.dim = dim
        self.num_heads = resolved_heads
        self.head_dim = dim // resolved_heads
        self.num_patterns = num_patterns
        self.num_steps = num_steps
        self.step_size = float(step_size)
        self.dropout = float(dropout)

        self.in_proj = nn.Linear(dim, dim * 2, bias=False)
        self.keys = nn.Parameter(
            torch.randn(resolved_heads, num_patterns, self.head_dim)
            * (self.head_dim ** -0.5)
        )
        self.values = nn.Parameter(
            torch.randn(resolved_heads, num_patterns, self.head_dim)
        )
        init_beta = max(_BETA_MIN, min(_BETA_MAX, math.sqrt(self.head_dim)))
        # 1D parameters (no weight decay under standard ndim >= 2 filters)
        self.log_beta = nn.Parameter(
            torch.full((resolved_heads,), math.log(init_beta))
        )
        if attn_sink:
            self.sink_logit: nn.Parameter | None = nn.Parameter(
                torch.zeros(resolved_heads)
            )
        else:
            self.register_parameter("sink_logit", None)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self._init_weights(residual_scale)

    def _init_weights(self, residual_scale: float) -> None:
        nn.init.normal_(self.in_proj.weight, std=1.0 / math.sqrt(self.dim))
        nn.init.normal_(
            self.out_proj.weight,
            std=(1.0 / math.sqrt(self.dim)) * residual_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        xi_raw, out_gate = self.in_proj(x).chunk(2, dim=-1)
        if len(lead) >= 1:
            b_outer = lead[0]
            seq_inner = math.prod(lead[1:]) if len(lead) > 1 else 1
        else:
            b_outer = 1
            seq_inner = 1

        if self.num_heads == 1:
            xi = xi_raw.reshape(b_outer, 1, seq_inner, self.head_dim)
        else:
            xi = (
                xi_raw.reshape(b_outer, seq_inner, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )  # [B_outer, H, L, d_h]
        xi = F.normalize(xi.float(), dim=-1, eps=1e-6).to(dtype=x.dtype)

        keys = F.normalize(self.keys.float(), dim=-1, eps=1e-6).to(dtype=x.dtype)
        beta = (
            self.log_beta.float()
            .clamp(math.log(_BETA_MIN), math.log(_BETA_MAX))
            .exp()
            .view(-1, 1, 1)
            .to(dtype=x.dtype)
        )
        keys_scaled = keys * beta  # [H, M, d_h]
        vals = self.values.to(dtype=x.dtype)

        has_sink = self.sink_logit is not None
        pad_d = 0
        if has_sink:
            # Pad head_dim by at least 1 up to a multiple of 8 for FlashAttention alignment:
            # Q is padded with 1.0, real keys with 0.0, and the appended sink key has
            # sink_logit in its last coordinate (and 0.0 elsewhere) with a zero sink value,
            # giving exact sink-softmax via fused SDPA without allocating [H, N, M] logits.
            pad_d = 8 - (self.head_dim % 8)
            d_ext = self.head_dim + pad_d
            k_real = F.pad(keys_scaled, (0, pad_d))
            k_sink = torch.cat(
                (
                    torch.zeros(
                        self.num_heads, 1, d_ext - 1, device=x.device, dtype=x.dtype
                    ),
                    self.sink_logit.to(dtype=x.dtype).view(self.num_heads, 1, 1),
                ),
                dim=-1,
            )
            k_all = (
                torch.cat((k_real, k_sink), dim=-2)
                .unsqueeze(0)
                .expand(b_outer, -1, -1, -1)
            )
            if self.num_steps > 1:
                # Store 1.0 in the last padded coordinate for real keys and 0.0
                # for the sink so SDPA returns exact non-sink mass (1 - w_sink)
                # in `retrieved[..., -1:]` with no extra kernel or allocation.
                real_flag = torch.ones(
                    self.num_heads, self.num_patterns, 1, device=x.device, dtype=x.dtype
                )
                v_keys_real = torch.cat(
                    (F.pad(keys, (0, pad_d - 1)), real_flag), dim=-1
                )
                v_keys = (
                    F.pad(v_keys_real, (0, 0, 0, 1))
                    .unsqueeze(0)
                    .expand(b_outer, -1, -1, -1)
                )
            else:
                v_keys = None
            v_vals = (
                F.pad(vals, (0, pad_d, 0, 1))
                .unsqueeze(0)
                .expand(b_outer, -1, -1, -1)
            )
        else:
            k_all = keys_scaled.unsqueeze(0).expand(b_outer, -1, -1, -1)
            v_keys = (
                keys.unsqueeze(0).expand(b_outer, -1, -1, -1)
                if self.num_steps > 1
                else None
            )
            v_vals = vals.unsqueeze(0).expand(b_outer, -1, -1, -1)

        dropout_p = self.dropout if self.training else 0.0
        out: torch.Tensor | None = None
        mass: torch.Tensor | float = 1.0
        for step in range(self.num_steps):
            if step == 0:
                xi_in = xi
            else:
                xi_unit = F.normalize(xi.float(), dim=-1, eps=1e-6)
                xi_in = (
                    (xi_unit * mass).to(dtype=x.dtype)
                    if has_sink
                    else xi_unit.to(dtype=x.dtype)
                )
            q_in = F.pad(xi_in, (0, pad_d), value=1.0) if has_sink else xi_in
            if step < self.num_steps - 1:
                assert v_keys is not None
                retrieved_ext = F.scaled_dot_product_attention(
                    q_in, k_all, v_keys, scale=1.0
                )
                if has_sink:
                    retrieved = retrieved_ext[..., : self.head_dim]
                    p_real = retrieved_ext[..., -1:].float()
                    mass = (
                        p_real
                        if self.step_size == 1.0
                        else mass + self.step_size * (p_real - mass)
                    )
                else:
                    retrieved = retrieved_ext
                if self.step_size == 1.0:
                    xi = retrieved
                else:
                    xi = xi_in + self.step_size * (retrieved - xi_in)
            else:
                out = F.scaled_dot_product_attention(
                    q_in, k_all, v_vals, dropout_p=dropout_p, scale=1.0
                )
                if has_sink:
                    out = out[..., : self.head_dim]

        assert out is not None
        out = out.transpose(1, 2).reshape(*lead, self.dim)
        out = self.out_proj(out) * torch.sigmoid(out_gate)
        return self.drop(out)
