from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding


def _resolve_tri_dim(
    dim: int,
    tri_dim: int | None,
    heads: int | None,
) -> int:
    if tri_dim is not None and heads is not None:
        if tri_dim != heads * 2:
            raise ValueError(
                f"In 2D Triangle-RoPE blocks, heads ({heads}) maps to tri_dim // 2; "
                f"got conflicting tri_dim={tri_dim} and heads={heads}."
            )
        return tri_dim
    if tri_dim is not None:
        return tri_dim
    if heads is not None:
        if heads <= 0:
            raise ValueError(f"heads must be positive, got {heads}.")
        return heads * 2
    return max(2, (dim // 2) // 2 * 2)


class TriangleRoPEBlock(nn.Module):
    """Base Bipartite 2D Time-Frequency Triangle Multiplicative Update with
    Real-Valued Relative-RoPE Contractions for `[B, T, F, D]` grids.

    Subclasses (`TimeTriangleRoPEBlock`, `WindowedTimeTriangleRoPEBlock`,
    `GlobalFreqTriangleRoPEBlock`, `WindowedFreqTriangleRoPEBlock`) each define a
    branch-free forward method so `torch.compile` keeps separate Dynamo caches per
    contraction type without hitting `recompile_limit` under activation
    checkpointing.

    Frequency-axis blocks require `freq_positions`: per-band log-frequency
    positions (semitones, see `rope.log_frequency_positions`). Half of the rotary
    pairs rotate by log-frequency (pitch-invariant harmonic ratios) and half by
    band index (linear harmonic-comb spacing).
    """

    def __init__(
        self,
        dim: int,
        tri_dim: int | None = 128,
        heads: int | None = None,
        axis: str = "time",
        ffn_mult: float = 3.0,
        rope_base: float = 10_000.0,
        freq_window: int = 0,
        window_shift: int = 0,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        freq_positions: torch.Tensor | None = None,
    ):
        super().__init__()
        resolved_tri_dim = _resolve_tri_dim(dim, tri_dim, heads)
        if axis not in ("time", "freq"):
            raise ValueError(f"axis must be 'time' or 'freq', got {axis!r}.")
        if resolved_tri_dim <= 0 or resolved_tri_dim % 2 != 0:
            raise ValueError(
                f"tri_dim must be a positive even integer, got {resolved_tri_dim}."
            )
        if freq_window < 0:
            raise ValueError(f"freq_window must be non-negative, got {freq_window}.")
        if window_shift < 0 or (freq_window > 0 and window_shift >= freq_window):
            raise ValueError(
                f"window_shift ({window_shift}) must be in [0, freq_window)."
            )
        self.dim = dim
        self.tri_dim = resolved_tri_dim
        self.heads = resolved_tri_dim // 2
        self.axis = axis
        self.freq_window = freq_window if axis == "freq" else 0
        self.window_shift = window_shift if self.freq_window > 0 else 0
        self.dropout = dropout
        ffn_hidden = max(64, int(round(dim * ffn_mult / 64.0)) * 64)

        self.tri_norm = nn.RMSNorm(dim)
        self.in_proj = nn.Linear(dim, resolved_tri_dim * 6 + dim, bias=False)
        self.a_norm = nn.RMSNorm(resolved_tri_dim)
        self.b_norm = nn.RMSNorm(resolved_tri_dim)
        self.c_norm = nn.RMSNorm(resolved_tri_dim)
        self.rope = RotaryEmbedding(resolved_tri_dim, base=rope_base)
        if axis == "freq":
            if freq_positions is None:
                raise ValueError(
                    "Frequency-axis Triangle-RoPE blocks require freq_positions "
                    "(per-band log-frequency positions)."
                )
            freq_cos, freq_sin = self.rope.freq_tables(torch.as_tensor(freq_positions))
            self.register_buffer("freq_cos", freq_cos, persistent=False)
            self.register_buffer("freq_sin", freq_sin, persistent=False)
            # 1D parameter (no weight decay) controlling softmax sharpness per channel
            self.freq_scale = nn.Parameter(torch.full((resolved_tri_dim // 2,), 2.0))
        else:
            self.register_buffer("freq_cos", None, persistent=False)
            self.register_buffer("freq_sin", None, persistent=False)
            self.register_parameter("freq_scale", None)
        self.tri_out_norm = nn.RMSNorm(resolved_tri_dim)
        self.tri_out = nn.Linear(resolved_tri_dim, dim, bias=False)

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
            std=(1.0 / math.sqrt(self.tri_dim)) * residual_scale,
        )
        ffn_out_std = (1.0 / math.sqrt(self.ffn_out.in_features)) * residual_scale
        nn.init.normal_(self.ffn_out.weight, std=ffn_out_std)

    def _project_abc(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.tri_norm(x)
        abc_proj, out_gate = self.in_proj(h).split(
            (self.tri_dim * 6, self.dim), dim=-1
        )
        a_val, a_gate, b_val, b_gate, c_val, c_gate = abc_proj.chunk(6, dim=-1)
        # Zero-centered tanh gates on (a, b) prevent positive DC bias under time sums
        a = self.a_norm(a_val) * torch.tanh(a_gate)
        b = self.b_norm(b_val) * torch.tanh(b_gate)
        c = self.c_norm(c_val) * torch.sigmoid(c_gate)
        return a, b, c, out_gate

    def _apply_freq_rope(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.freq_cos is not None and self.freq_sin is not None
        return self.rope.apply_freq_tables(a, b, self.freq_cos, self.freq_sin)

    def _finish_block(
        self,
        x: torch.Tensor,
        o_0: torch.Tensor,
        o_1: torch.Tensor,
        out_gate: torch.Tensor,
    ) -> torch.Tensor:
        tri = torch.cat((o_0, o_1), dim=1).permute(0, 2, 3, 1).contiguous()
        tri = self.tri_out(self.tri_out_norm(tri)) * torch.sigmoid(out_gate)
        x = x + self.drop(tri)

        h = self.ffn_norm(x)
        ffn_gate, ffn_val = self.ffn_in(h).chunk(2, dim=-1)
        ffn = F.silu(ffn_gate) * ffn_val
        return x + self.drop(self.ffn_out(self.drop(ffn)))


class TimeTriangleRoPEBlock(TriangleRoPEBlock):
    """Full-spectrum time-to-time real-RoPE contraction in O(T * F^2):

    Non-causal (`causal=False`):
        K_r[f', f] = (1/T) * sum_{t'} b_r[t', f'] * c[t', f]
        K_i[f', f] = (1/T) * sum_{t'} b_i[t', f'] * c[t', f]
    with phase-preserving joint RMS normalization on (K_r, K_i) before
        O[t, f] = (1/sqrt(F)) * sum_{f'} (a_r[t, f'] * K_r[f', f] + a_i[t, f'] * K_i[f', f]).

    Causal (`causal=True`):
        Computes the exact prefix contraction over t' <= t in chunks of `chunk_size`.
    """

    def __init__(
        self,
        dim: int,
        tri_dim: int | None = 128,
        heads: int | None = None,
        causal: bool = False,
        chunk_size: int = 32,
        ffn_mult: float = 3.0,
        rope_base: float = 10_000.0,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
    ):
        super().__init__(
            dim=dim,
            tri_dim=tri_dim,
            heads=heads,
            axis="time",
            ffn_mult=ffn_mult,
            rope_base=rope_base,
            dropout=dropout,
            residual_scale=residual_scale,
        )
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
        self.causal = causal
        self.chunk_size = chunk_size

    def _causal_forward(
        self,
        a_r: torch.Tensor,
        a_i: torch.Tensor,
        b_r: torch.Tensor,
        b_i: torch.Tensor,
        c_cat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, m_ch, frames, bands = a_r.shape
        two_f = c_cat.shape[-1]
        inv_sqrt_f = 1.0 / math.sqrt(bands)

        s_r = torch.zeros(
            batch, m_ch, bands, two_f, device=a_r.device, dtype=torch.float32
        )
        s_i = torch.zeros_like(s_r)
        outputs: list[torch.Tensor] = []

        for start in range(0, frames, self.chunk_size):
            end = min(start + self.chunk_size, frames)
            chunk_len = end - start

            ar_c = a_r[:, :, start:end, :].float()
            ai_c = a_i[:, :, start:end, :].float()
            br_c = b_r[:, :, start:end, :].float()
            bi_c = b_i[:, :, start:end, :].float()
            c_c = c_cat[:, :, start:end, :].float()

            delta_r = br_c.unsqueeze(-1) * c_c.unsqueeze(-2)
            delta_i = bi_c.unsqueeze(-1) * c_c.unsqueeze(-2)

            cum_r = s_r.unsqueeze(2) + torch.cumsum(delta_r, dim=2)
            cum_i = s_i.unsqueeze(2) + torch.cumsum(delta_i, dim=2)
            s_r = cum_r[:, :, -1]
            s_i = cum_i[:, :, -1]

            denom = torch.arange(
                start + 1, end + 1, device=a_r.device, dtype=torch.float32
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
            ) * inv_sqrt_f
            outputs.append(o_c.to(dtype=c_cat.dtype))

        o_cat = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=2)
        return o_cat.chunk(2, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _batch, frames, bands, _dim = x.shape
        a, b, c, out_gate = self._project_abc(x)

        a_r, a_i, b_r, b_i = self.rope.apply_time(a, b)
        c_0, c_1 = c.chunk(2, dim=-1)

        a_r = a_r.permute(0, 3, 1, 2)
        a_i = a_i.permute(0, 3, 1, 2)
        b_r = b_r.permute(0, 3, 1, 2)
        b_i = b_i.permute(0, 3, 1, 2)
        c_0 = c_0.permute(0, 3, 1, 2)
        c_1 = c_1.permute(0, 3, 1, 2)

        c_cat = torch.cat((c_0, c_1), dim=-1)  # [B, M, T, 2*F]
        if self.causal:
            o_0, o_1 = self._causal_forward(a_r, a_i, b_r, b_i, c_cat)
            return self._finish_block(x, o_0, o_1, out_gate)

        inv_t = 1.0 / frames
        inv_sqrt_f = 1.0 / math.sqrt(bands)
        k_r = torch.matmul(b_r.transpose(-2, -1), c_cat) * inv_t
        k_i = torch.matmul(b_i.transpose(-2, -1), c_cat) * inv_t
        k_rms = (
            (k_r.float().square() + k_i.float().square())
            .mean(dim=-2, keepdim=True)
            .add(1e-6)
            .sqrt()
            .to(dtype=k_r.dtype)
        )
        k_r = k_r / k_rms
        k_i = k_i / k_rms
        o_cat = (torch.matmul(a_r, k_r) + torch.matmul(a_i, k_i)) * inv_sqrt_f
        o_0, o_1 = o_cat.chunk(2, dim=-1)
        return self._finish_block(x, o_0, o_1, out_gate)


class WindowedTimeTriangleRoPEBlock(TriangleRoPEBlock):
    """Local (optionally half-shifted) windowed time-to-time real-RoPE contraction.

    Frames are tiled into non-overlapping windows of `time_window` frames (after a
    left pad of `window_shift`). Each window w gets its own transfer matrix
        K_r^w[f', f] = (1/|w|) * sum_{t' in w} b_r[t', f'] * c[t', f]
        K_i^w[f', f] = (1/|w|) * sum_{t' in w} b_i[t', f'] * c[t', f]
    (joint RMS-normalized as in `TimeTriangleRoPEBlock`) and
        O[t, f] = (1/sqrt(F)) * sum_{f'} (a_r[t, f'] K_r^w[f', f] + a_i[t, f'] K_i^w[f', f])
    for t in w. Same O(T * F^2) cost and parameters as the global block, but
    short-range phrasing is not averaged against the whole chunk. Alternating
    the shift across layers moves window seams.
    """

    def __init__(
        self,
        dim: int,
        tri_dim: int | None = 128,
        heads: int | None = None,
        ffn_mult: float = 3.0,
        rope_base: float = 10_000.0,
        time_window: int = 64,
        window_shift: int = 0,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
    ):
        super().__init__(
            dim=dim,
            tri_dim=tri_dim,
            heads=heads,
            axis="time",
            ffn_mult=ffn_mult,
            rope_base=rope_base,
            dropout=dropout,
            residual_scale=residual_scale,
        )
        if time_window <= 0:
            raise ValueError(f"time_window must be positive, got {time_window}.")
        if not 0 <= window_shift < time_window:
            raise ValueError(
                f"window_shift ({window_shift}) must be in [0, time_window)."
            )
        self.time_window = time_window
        self.time_window_shift = window_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, bands, _dim = x.shape
        a, b, c, out_gate = self._project_abc(x)

        a_r, a_i, b_r, b_i = self.rope.apply_time(a, b)
        c_0, c_1 = c.chunk(2, dim=-1)

        a_r = a_r.permute(0, 3, 1, 2)
        a_i = a_i.permute(0, 3, 1, 2)
        b_r = b_r.permute(0, 3, 1, 2)
        b_i = b_i.permute(0, 3, 1, 2)
        c_cat = torch.cat((c_0.permute(0, 3, 1, 2), c_1.permute(0, 3, 1, 2)), dim=-1)

        win = self.time_window
        shift = self.time_window_shift
        pad_right = (-(frames + shift)) % win
        pad_spec = (0, 0, shift, pad_right)
        a_r = F.pad(a_r, pad_spec)
        a_i = F.pad(a_i, pad_spec)
        b_r = F.pad(b_r, pad_spec)
        b_i = F.pad(b_i, pad_spec)
        c_cat = F.pad(c_cat, pad_spec)

        padded_t = a_r.shape[2]
        num_win = padded_t // win
        m_ch = a_r.shape[1]
        two_f = c_cat.shape[-1]

        a_rw = a_r.reshape(batch, m_ch, num_win, win, bands)
        a_iw = a_i.reshape(batch, m_ch, num_win, win, bands)
        b_rw = b_r.reshape(batch, m_ch, num_win, win, bands)
        b_iw = b_i.reshape(batch, m_ch, num_win, win, bands)
        c_w = c_cat.reshape(batch, m_ch, num_win, win, two_f)

        # Number of real (unpadded) frames per window, for an unbiased mean.
        win_start = torch.arange(num_win, device=x.device) * win - shift
        valid = (
            torch.clamp(win_start + win, max=frames) - torch.clamp(win_start, min=0)
        ).clamp_min(1)
        inv_count = (1.0 / valid.to(torch.float32)).view(1, 1, num_win, 1, 1)
        inv_count = inv_count.to(dtype=c_w.dtype)

        k_r = torch.matmul(b_rw.transpose(-2, -1), c_w) * inv_count
        k_i = torch.matmul(b_iw.transpose(-2, -1), c_w) * inv_count
        k_rms = (
            (k_r.float().square() + k_i.float().square())
            .mean(dim=-2, keepdim=True)
            .add(1e-6)
            .sqrt()
            .to(dtype=k_r.dtype)
        )
        k_r = k_r / k_rms
        k_i = k_i / k_rms

        inv_sqrt_f = 1.0 / math.sqrt(bands)
        o_w = (torch.matmul(a_rw, k_r) + torch.matmul(a_iw, k_i)) * inv_sqrt_f
        o_cat = o_w.reshape(batch, m_ch, padded_t, two_f)[:, :, shift : shift + frames, :]
        o_0, o_1 = o_cat.chunk(2, dim=-1)
        return self._finish_block(x, o_0, o_1, out_gate)


class GlobalFreqTriangleRoPEBlock(TriangleRoPEBlock):
    """Global chunk-wide real-RoPE band-to-band softmax affinity:
        M[f, f'] = Re(mean_{t'} (R_f a[t', f]) * conj(R_{f'} b[t', f']))
        A[f, f'] = softmax_{f'}(RMSNorm_{f'}(M[f, f']) * freq_scale)
        O[t, f]  = sum_{f'} A[f, f'] * c[t, f'].
    """

    def __init__(
        self,
        dim: int,
        tri_dim: int | None = 128,
        heads: int | None = None,
        ffn_mult: float = 3.0,
        rope_base: float = 1_000.0,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        freq_positions: torch.Tensor | None = None,
    ):
        super().__init__(
            dim=dim,
            tri_dim=tri_dim,
            heads=heads,
            axis="freq",
            ffn_mult=ffn_mult,
            rope_base=rope_base,
            freq_window=0,
            window_shift=0,
            dropout=dropout,
            residual_scale=residual_scale,
            freq_positions=freq_positions,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _batch, frames, _bands, _dim = x.shape
        a, b, c, out_gate = self._project_abc(x)

        a_r, a_i, b_r, b_i = self._apply_freq_rope(a, b)
        c_0, c_1 = c.chunk(2, dim=-1)

        a_r = a_r.permute(0, 3, 1, 2)
        a_i = a_i.permute(0, 3, 1, 2)
        b_r = b_r.permute(0, 3, 1, 2)
        b_i = b_i.permute(0, 3, 1, 2)
        c_0 = c_0.permute(0, 3, 1, 2)
        c_1 = c_1.permute(0, 3, 1, 2)

        assert self.freq_scale is not None
        scale = self.freq_scale.float().clamp(0.1, 20.0).view(1, -1, 1, 1)
        m = (
            torch.matmul(a_r.transpose(-2, -1), b_r)
            + torch.matmul(a_i.transpose(-2, -1), b_i)
        ) * (1.0 / frames)
        m_f = m.float()
        m_rms = m_f.square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        attn = F.softmax((m_f / m_rms) * scale, dim=-1).to(dtype=c_0.dtype)
        c_freq = torch.cat((c_0, c_1), dim=-2)
        o_freq = torch.matmul(c_freq, attn.transpose(-2, -1))
        o_0, o_1 = o_freq.chunk(2, dim=-2)
        return self._finish_block(x, o_0, o_1, out_gate)


class WindowedFreqTriangleRoPEBlock(TriangleRoPEBlock):
    """Local time-windowed (optionally half-shifted) real-RoPE band-to-band
    softmax affinity for tracking fast-moving harmonics and melodies.
    """

    def __init__(
        self,
        dim: int,
        tri_dim: int | None = 128,
        heads: int | None = None,
        ffn_mult: float = 3.0,
        rope_base: float = 1_000.0,
        freq_window: int = 32,
        window_shift: int = 0,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        freq_positions: torch.Tensor | None = None,
    ):
        super().__init__(
            dim=dim,
            tri_dim=tri_dim,
            heads=heads,
            axis="freq",
            ffn_mult=ffn_mult,
            rope_base=rope_base,
            freq_window=freq_window,
            window_shift=window_shift,
            dropout=dropout,
            residual_scale=residual_scale,
            freq_positions=freq_positions,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, bands, _dim = x.shape
        a, b, c, out_gate = self._project_abc(x)

        a_r, a_i, b_r, b_i = self._apply_freq_rope(a, b)
        c_0, c_1 = c.chunk(2, dim=-1)

        a_r = a_r.permute(0, 3, 1, 2)
        a_i = a_i.permute(0, 3, 1, 2)
        b_r = b_r.permute(0, 3, 1, 2)
        b_i = b_i.permute(0, 3, 1, 2)
        c_0 = c_0.permute(0, 3, 1, 2)
        c_1 = c_1.permute(0, 3, 1, 2)

        assert self.freq_scale is not None
        scale = self.freq_scale.float().clamp(0.1, 20.0)
        win = self.freq_window
        shift = self.window_shift
        pad_right = (-(frames + shift)) % win
        pad_spec = (0, 0, shift, pad_right)
        a_r = F.pad(a_r, pad_spec)
        a_i = F.pad(a_i, pad_spec)
        b_r = F.pad(b_r, pad_spec)
        b_i = F.pad(b_i, pad_spec)
        c_0 = F.pad(c_0, pad_spec)
        c_1 = F.pad(c_1, pad_spec)

        padded_t = a_r.shape[2]
        num_win = padded_t // win
        m_ch = a_r.shape[1]

        a_rw = a_r.reshape(batch, m_ch, num_win, win, bands)
        a_iw = a_i.reshape(batch, m_ch, num_win, win, bands)
        b_rw = b_r.reshape(batch, m_ch, num_win, win, bands)
        b_iw = b_i.reshape(batch, m_ch, num_win, win, bands)

        m = (
            torch.matmul(a_rw.transpose(-2, -1), b_rw)
            + torch.matmul(a_iw.transpose(-2, -1), b_iw)
        ) * (1.0 / win)
        m_f = m.float()
        m_rms = m_f.square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        attn = F.softmax(
            (m_f / m_rms) * scale.view(1, -1, 1, 1, 1), dim=-1
        ).to(dtype=c_0.dtype)

        c_w = torch.cat(
            (
                c_0.reshape(batch, m_ch, num_win, win, bands),
                c_1.reshape(batch, m_ch, num_win, win, bands),
            ),
            dim=-2,
        )
        o_w = torch.matmul(c_w, attn.transpose(-2, -1))
        o_0w, o_1w = o_w.chunk(2, dim=-2)
        o_0 = o_0w.reshape(batch, m_ch, padded_t, bands)[
            :, :, shift : shift + frames, :
        ]
        o_1 = o_1w.reshape(batch, m_ch, padded_t, bands)[
            :, :, shift : shift + frames, :
        ]
        return self._finish_block(x, o_0, o_1, out_gate)


def make_freq_triangle_block(
    dim: int,
    tri_dim: int | None = 128,
    heads: int | None = None,
    ffn_mult: float = 3.0,
    rope_base: float = 1_000.0,
    freq_window: int = 0,
    window_shift: int = 0,
    dropout: float = 0.0,
    residual_scale: float = 1.0,
    freq_positions: torch.Tensor | None = None,
) -> TriangleRoPEBlock:
    if freq_window > 0:
        return WindowedFreqTriangleRoPEBlock(
            dim=dim,
            tri_dim=tri_dim,
            heads=heads,
            ffn_mult=ffn_mult,
            rope_base=rope_base,
            freq_window=freq_window,
            window_shift=window_shift,
            dropout=dropout,
            residual_scale=residual_scale,
            freq_positions=freq_positions,
        )
    return GlobalFreqTriangleRoPEBlock(
        dim=dim,
        tri_dim=tri_dim,
        heads=heads,
        ffn_mult=ffn_mult,
        rope_base=rope_base,
        dropout=dropout,
        residual_scale=residual_scale,
        freq_positions=freq_positions,
    )


def make_time_triangle_block(
    dim: int,
    tri_dim: int | None = 128,
    heads: int | None = None,
    causal: bool = False,
    chunk_size: int = 32,
    ffn_mult: float = 3.0,
    rope_base: float = 10_000.0,
    time_window: int = 0,
    window_shift: int = 0,
    dropout: float = 0.0,
    residual_scale: float = 1.0,
) -> TriangleRoPEBlock:
    """`time_window > 0` builds a local windowed (non-causal) time contraction;
    `time_window == 0` builds the chunk-wide (or causal prefix) contraction."""
    if time_window > 0:
        if causal:
            raise ValueError("Windowed time contraction is non-causal; use time_window=0.")
        return WindowedTimeTriangleRoPEBlock(
            dim=dim,
            tri_dim=tri_dim,
            heads=heads,
            ffn_mult=ffn_mult,
            rope_base=rope_base,
            time_window=time_window,
            window_shift=window_shift,
            dropout=dropout,
            residual_scale=residual_scale,
        )
    return TimeTriangleRoPEBlock(
        dim=dim,
        tri_dim=tri_dim,
        heads=heads,
        causal=causal,
        chunk_size=chunk_size,
        ffn_mult=ffn_mult,
        rope_base=rope_base,
        dropout=dropout,
        residual_scale=residual_scale,
    )
