from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._compile import compile_unit, run_unit
from .blocks_2d import TimeTriangleRoPEBlock, make_freq_triangle_block


class ConvNeXtV2TemporalMixer(nn.Module):
    """ConvNeXt-V2-style local temporal mixer with RMSNorm, SwiGLU gating,
    and Global Response Normalization (GRN).

    Expects `[N, frames, dim]` (e.g., `[B * bands, frames, dim]`).
    When `causal=True`, applies left-only padding on the depthwise conv so
    frame `t` never sees frames `> t`.

    `frame_mask` (`[N, frames]`, 1 = real frame) zeroes padded frames before the
    depthwise conv and excludes them from the (non-causal) GRN statistics.
    """

    def __init__(
        self,
        dim: int,
        ffn_mult: float = 2.0,
        kernel_size: int = 7,
        causal: bool = False,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        hidden_dim = max(64, int(round(dim * ffn_mult / 64.0)) * 64)
        self.kernel_size = kernel_size
        self.causal = causal
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=0 if causal else kernel_size // 2,
            groups=dim,
        )
        self.norm = nn.RMSNorm(dim)
        self.pw_in = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.grn_gamma = nn.Parameter(torch.zeros(hidden_dim))
        self.grn_beta = nn.Parameter(torch.zeros(hidden_dim))
        self.pw_out = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        nn.init.zeros_(self.dwconv.bias)
        nn.init.normal_(self.pw_in.weight, std=1.0 / math.sqrt(dim))
        nn.init.normal_(
            self.pw_out.weight,
            std=(1.0 / math.sqrt(hidden_dim)) * residual_scale,
        )

    def forward(
        self, x: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        valid = None
        h_in = x
        if frame_mask is not None:
            valid = frame_mask.to(device=x.device, dtype=torch.float32)[:, :, None]
            h_in = x * valid.to(dtype=x.dtype)
        h_in = h_in.transpose(1, 2)
        if self.causal:
            h_in = F.pad(h_in, (self.kernel_size - 1, 0))
        h = self.dwconv(h_in).transpose(1, 2)
        h = self.norm(h)
        gate, val = self.pw_in(h).chunk(2, dim=-1)
        h = F.silu(gate) * val
        if self.causal:
            # Causal per-frame RMS over channels (avoids future leakage across time)
            gx = h.float().square().add(1e-6).sqrt()
        elif valid is None:
            gx = h.float().square().mean(dim=1, keepdim=True).add(1e-6).sqrt()
        else:
            count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            gx = (
                (h.float().square() * valid).sum(dim=1, keepdim=True) / count
            ).add(1e-6).sqrt()
        nx = (gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)).to(dtype=h.dtype)
        h = h + self.grn_gamma.to(dtype=h.dtype) * (h * nx) + self.grn_beta.to(
            dtype=h.dtype
        )
        return x + self.drop(self.pw_out(self.drop(h)))


class FineRateBoundaryBlock(nn.Module):
    """Fine-rate boundary block (PreBlock / PostBlock) for 2D grids:
    ConvNeXt-V2 temporal mixer -> time-axis Triangle-RoPE -> freq-axis Triangle-RoPE.
    """

    def __init__(
        self,
        dim: int,
        tri_dim: int | None = 128,
        heads: int | None = None,
        causal: bool = False,
        ffn_mult: float = 3.0,
        freq_window: int = 32,
        freq_window_shift: int = 0,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        freq_positions: torch.Tensor | None = None,
        chunk_size: int = 32,
        soft_norm: bool = True,
        norm_floor_init: float = 0.5,
        attn_sink: bool = True,
        causal_decay: bool = True,
    ):
        super().__init__()
        self.temporal_mixer = ConvNeXtV2TemporalMixer(
            dim=dim,
            ffn_mult=min(2.0, ffn_mult),
            kernel_size=7,
            causal=causal,
            dropout=dropout,
            residual_scale=residual_scale,
        )
        self.time_layer = TimeTriangleRoPEBlock(
            dim=dim,
            tri_dim=tri_dim,
            heads=heads,
            causal=causal,
            chunk_size=chunk_size,
            ffn_mult=ffn_mult,
            rope_base=10_000.0,
            dropout=dropout,
            residual_scale=residual_scale,
            soft_norm=soft_norm,
            norm_floor_init=norm_floor_init,
            causal_decay=causal_decay,
        )
        self.freq_layer = make_freq_triangle_block(
            dim=dim,
            tri_dim=tri_dim,
            heads=heads,
            ffn_mult=ffn_mult,
            rope_base=1_000.0,
            freq_window=freq_window,
            window_shift=freq_window_shift,
            dropout=dropout,
            residual_scale=residual_scale,
            freq_positions=freq_positions,
            causal=causal,
            chunk_size=chunk_size,
            soft_norm=soft_norm,
            norm_floor_init=norm_floor_init,
            attn_sink=attn_sink,
            causal_decay=causal_decay,
        )

    def compile_layers(self, mode: str = "default") -> None:
        compile_unit(self.temporal_mixer, mode=mode)
        compile_unit(self.time_layer, mode=mode)
        compile_unit(self.freq_layer, mode=mode)

    @staticmethod
    def _run_unit(
        module: nn.Module,
        x: torch.Tensor,
        use_checkpoint: bool,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return run_unit(module, use_checkpoint, x, frame_mask)

    def forward(
        self,
        x: torch.Tensor,
        should_checkpoint: bool = False,
        keep_t: torch.Tensor | float | None = None,
        keep_f: torch.Tensor | float | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, frames, bands, dim = x.shape
        t_in = x.permute(0, 2, 1, 3).reshape(batch * bands, frames, dim)
        t_mask = (
            None
            if frame_mask is None
            else frame_mask.repeat_interleave(bands, dim=0)
        )
        t_mixed = self._run_unit(
            self.temporal_mixer, t_in, should_checkpoint, t_mask
        )
        x_mixed = t_mixed.reshape(batch, bands, frames, dim).permute(0, 2, 1, 3)

        if keep_t is not None and isinstance(keep_t, float) and keep_t == 0.0:
            x = x_mixed
        else:
            t_out = self._run_unit(
                self.time_layer, x_mixed, should_checkpoint, frame_mask
            )
            x = t_out if keep_t is None else x_mixed + (t_out - x_mixed) * keep_t

        if keep_f is not None and isinstance(keep_f, float) and keep_f == 0.0:
            return x
        f_out = self._run_unit(self.freq_layer, x, should_checkpoint, frame_mask)
        return f_out if keep_f is None else x + (f_out - x) * keep_f


class GEGLUFusion(nn.Module):
    """Orthogonal Discrepancy GEGLU Gated Residual Fusion Module.

    Projects normalized skip features `X'` onto the subspace orthogonal to the
    normalized deep representation `Y'` in fp32 before gated fusion:
        S = RMSNorm(X'),  C = RMSNorm(Y')
        S_ortho = S - <S, C_hat> C_hat
        Y_fuse = Y' + W_o(GELU(W_g [S_ortho, C]) * W_v [S_ortho, C])
    """

    def __init__(
        self,
        dim: int,
        ffn_mult: float = 2.0,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        hidden_dim = max(64, int(round(dim * ffn_mult / 64.0)) * 64)
        self.skip_norm = nn.RMSNorm(dim)
        self.coarse_norm = nn.RMSNorm(dim)
        self.wgv = nn.Linear(dim * 2, hidden_dim * 2, bias=False)
        self.wo = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        nn.init.normal_(self.wgv.weight, std=1.0 / math.sqrt(dim * 2))
        nn.init.normal_(
            self.wo.weight,
            std=(1.0 / math.sqrt(hidden_dim)) * residual_scale,
        )

    def forward(
        self,
        x_skip: torch.Tensor,
        y_coarse: torch.Tensor,
    ) -> torch.Tensor:
        s_f = self.skip_norm(x_skip).float()
        c = self.coarse_norm(y_coarse)
        c_dir = F.normalize(c.float(), dim=-1, eps=1e-6)
        s_ortho = (
            s_f - (s_f * c_dir).sum(dim=-1, keepdim=True) * c_dir
        ).to(dtype=c.dtype)
        y_cat = torch.cat((s_ortho, c), dim=-1)
        wg, wv = self.wgv(y_cat).chunk(2, dim=-1)
        gated = F.gelu(wg) * wv
        return y_coarse + self.drop(self.wo(self.drop(gated)))
