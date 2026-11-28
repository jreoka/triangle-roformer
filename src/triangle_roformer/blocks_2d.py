from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hopfield import HopfieldFFN, _resolve_num_heads
from .rope import RotaryEmbedding


_FLOOR_MIN = 1e-3
_FLOOR_MAX = 10.0


def _decay_logit_init(
    channels: int, min_half_life: float, max_half_life: float
) -> torch.Tensor:
    """Per-channel decay logits whose half-lives (in frames) are log-spaced
    between `min_half_life` and `max_half_life` (multi-scale memory)."""
    half_life = torch.logspace(
        math.log10(min_half_life), math.log10(max_half_life), channels
    )
    gamma = torch.pow(2.0, -1.0 / half_life)
    return torch.log(gamma) - torch.log1p(-gamma)


def _decay_weights(
    log_gamma: torch.Tensor | None,
    length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Intra-chunk decay weights for a decayed prefix sum.

    Returns `w [M', L, L]` with `w[m, j, i] = gamma_m^(j - i)` for `i <= j` (else 0)
    and `carry [M', L]` with `carry[m, j] = gamma_m^(j + 1)` (weight of the state
    carried in from previous chunks). `log_gamma=None` means no decay (`M' = 1`),
    which reduces to a plain cumulative sum.
    """
    idx = torch.arange(length, device=device, dtype=torch.float32)
    lag = idx[:, None] - idx[None, :]
    lower = lag >= 0
    if log_gamma is None:
        return lower.float()[None], torch.ones(1, length, device=device)
    lg = log_gamma.float().to(device)
    w = torch.where(
        lower[None], torch.exp(lg.view(-1, 1, 1) * lag.clamp_min(0.0)[None]), 0.0
    )
    carry = torch.exp(lg.view(-1, 1) * (idx + 1.0)[None])
    return w, carry


def _decayed_chunk_scan(
    delta: torch.Tensor,
    valid: torch.Tensor,
    w: torch.Tensor,
    carry: torch.Tensor,
    state: torch.Tensor,
    count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decayed causal prefix sums over one chunk.

    delta: `[B, M, L, *rest]` per-frame terms (fp32), valid: `[B, L]` frame weights,
    state: `[B, M, *rest]` carried sum, count: `[B, M]` carried weight.
    Returns `cum [B, M, L, *rest]` and `n [B, M, L]` (decayed valid-frame counts).
    """
    batch, m_ch, length = delta.shape[:3]
    rest = delta.shape[3:]
    w = w.expand(m_ch, length, length)
    flat = delta.reshape(batch, m_ch, length, -1)
    cum = torch.einsum("mji,bmir->bmjr", w, flat)
    cum = cum + carry[None, :, :, None] * state.reshape(batch, m_ch, 1, -1)
    n = torch.einsum("mji,bi->bmj", w, valid) + carry[None] * count[:, :, None]
    return cum.reshape(batch, m_ch, length, *rest), n


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

    Separation-oriented options:
      - `soft_norm`: transfer matrices / affinity logits are normalized by
        `sqrt(rms^2 + floor^2 / n)` instead of by their own RMS, where `n` is the
        number of frames pooled and `floor` is a learnable per-channel parameter
        (`norm_floor_init`). The sampling noise of a mean over `n` frames scales
        like `1 / sqrt(n)`, so the floor sits at the noise level for every window
        size / chunk length: weak (noise-level) correlations stay weak instead of
        being rescaled to the same magnitude as strong ones.
      - `attn_sink` (frequency axis): a learnable per-channel null logit with a
        zero value, so a band can attend to "nothing" (e.g. when the target
        source is absent) instead of being forced to average other bands.
      - `frame_mask` (forward arg, `[B, T]`, 1 = real frame): padded frames are
        excluded from all time-pooled statistics.
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
        soft_norm: bool = True,
        norm_floor_init: float = 0.5,
        attn_sink: bool = True,
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
        if soft_norm and not _FLOOR_MIN <= norm_floor_init <= _FLOOR_MAX:
            raise ValueError(
                f"norm_floor_init must be in [{_FLOOR_MIN}, {_FLOOR_MAX}], "
                f"got {norm_floor_init}."
            )
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
        if soft_norm:
            self.log_norm_floor = nn.Parameter(
                torch.full((resolved_tri_dim // 2,), math.log(norm_floor_init))
            )
        else:
            self.register_parameter("log_norm_floor", None)
        if axis == "freq" and attn_sink:
            self.sink_logit = nn.Parameter(torch.zeros(resolved_tri_dim // 2))
        else:
            self.register_parameter("sink_logit", None)
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

    @staticmethod
    def _mask_abc(
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        frame_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Zero (a, b, c) on padded frames; returns them plus `valid [B, T]` (fp32)."""
        if frame_mask is None:
            valid = torch.ones(a.shape[0], a.shape[1], device=a.device)
            return a, b, c, valid
        valid = frame_mask.to(device=a.device, dtype=torch.float32)
        m = valid.to(dtype=a.dtype)[:, :, None, None]
        return a * m, b * m, c * m, valid

    def _channel_view(self, param: torch.Tensor, ndim: int) -> torch.Tensor:
        """Reshape a per-channel `[M]` parameter to broadcast over dim 1 of an `ndim` tensor."""
        return param.float().view(1, -1, *([1] * (ndim - 2)))

    def _floor_sq(
        self, ndim: int, inv_count: torch.Tensor
    ) -> torch.Tensor | float:
        """Squared normalization floor `floor^2 / n` (or the plain eps without
        `soft_norm`). `inv_count` (fp32, `1 / n`) must broadcast to the stats."""
        if self.log_norm_floor is None:
            return 1e-6
        floor = self.log_norm_floor.float().exp().clamp(_FLOOR_MIN, _FLOOR_MAX)
        return self._channel_view(floor.square(), ndim) * inv_count + 1e-6

    def _normalize_kernel(
        self, k_r: torch.Tensor, k_i: torch.Tensor, inv_count: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Phase-preserving joint normalization of `(K_r, K_i)` over source bands
        (dim -2). With `soft_norm`, divides by `sqrt(ms + floor^2 / n)` so the
        kernel keeps its correlation strength when it is near the noise level."""
        ms = (k_r.float().square() + k_i.float().square()).mean(dim=-2, keepdim=True)
        scale = torch.rsqrt(ms + self._floor_sq(ms.ndim, inv_count))
        return k_r * scale.to(dtype=k_r.dtype), k_i * scale.to(dtype=k_i.dtype)

    def _freq_attention(self, m: torch.Tensor, inv_count: torch.Tensor) -> torch.Tensor:
        """Softmax over the last (source band) axis of a `[B, M, ..., F, F]`
        affinity, with optional soft normalization and a null (sink) logit."""
        assert self.freq_scale is not None
        m_f = m.float()
        ms = m_f.square().mean(dim=-1, keepdim=True)
        scale = self._channel_view(self.freq_scale.clamp(0.1, 20.0), m_f.ndim)
        logits = m_f * torch.rsqrt(ms + self._floor_sq(m_f.ndim, inv_count)) * scale
        if self.sink_logit is None:
            return F.softmax(logits, dim=-1)
        sink = self._channel_view(self.sink_logit, m_f.ndim).expand(
            *logits.shape[:-1], 1
        )
        return F.softmax(torch.cat((logits, sink), dim=-1), dim=-1)[..., :-1]

    def _apply_freq_rope(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.freq_cos is not None and self.freq_sin is not None
        return self.rope.apply_freq_tables(a, b, self.freq_cos, self.freq_sin)

    @staticmethod
    def _window_inv_counts(
        valid: torch.Tensor, shift: int, pad_right: int, win: int
    ) -> torch.Tensor:
        """`[B, 1, num_win, 1, 1]` (fp32) reciprocal count of real frames per window."""
        valid_p = F.pad(valid, (shift, pad_right))
        counts = valid_p.reshape(valid.shape[0], -1, win).sum(dim=-1).clamp_min(1.0)
        return (1.0 / counts)[:, None, :, None, None]

    @staticmethod
    def _global_inv_count(valid: torch.Tensor) -> torch.Tensor:
        """`[B, 1, 1, 1]` (fp32) reciprocal count of real frames in the chunk."""
        return (1.0 / valid.sum(dim=1).clamp_min(1.0)).view(-1, 1, 1, 1)

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
    with phase-preserving joint (soft) RMS normalization on (K_r, K_i) before
        O[t, f] = (1/sqrt(F)) * sum_{f'} (a_r[t, f'] * K_r[f', f] + a_i[t, f'] * K_i[f', f]).

    Causal (`causal=True`):
        Computes the exact prefix contraction over t' <= t in chunks of `chunk_size`.
        With `causal_decay=True` (default) the prefix is an exponentially decayed
        mean with a learnable per-channel decay (half-lives initialized log-spaced
        over `decay_half_life` frames), so recent frames are not diluted by the
        whole history; with `causal_decay=False` it is the plain running mean.
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
        soft_norm: bool = True,
        norm_floor_init: float = 0.5,
        causal_decay: bool = True,
        decay_half_life: tuple[float, float] = (4.0, 4096.0),
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
            soft_norm=soft_norm,
            norm_floor_init=norm_floor_init,
        )
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
        self.causal = causal
        self.chunk_size = chunk_size
        if causal and causal_decay:
            self.time_decay_logit = nn.Parameter(
                _decay_logit_init(self.heads, *decay_half_life)
            )
        else:
            self.register_parameter("time_decay_logit", None)

    def _causal_forward(
        self,
        a_r: torch.Tensor,
        a_i: torch.Tensor,
        b_r: torch.Tensor,
        b_i: torch.Tensor,
        c_cat: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, m_ch, frames, bands = a_r.shape
        two_f = c_cat.shape[-1]
        inv_sqrt_f = 1.0 / math.sqrt(bands)
        log_gamma = (
            None
            if self.time_decay_logit is None
            else F.logsigmoid(self.time_decay_logit.float())
        )

        state = torch.zeros(
            batch, m_ch, bands, 2 * two_f, device=a_r.device, dtype=torch.float32
        )
        count = torch.zeros(batch, m_ch, device=a_r.device, dtype=torch.float32)
        outputs: list[torch.Tensor] = []

        for start in range(0, frames, self.chunk_size):
            end = min(start + self.chunk_size, frames)
            w, carry = _decay_weights(log_gamma, end - start, a_r.device)

            ar_c = a_r[:, :, start:end, :].float()
            ai_c = a_i[:, :, start:end, :].float()
            br_c = b_r[:, :, start:end, :].float()
            bi_c = b_i[:, :, start:end, :].float()
            c_c = c_cat[:, :, start:end, :].float()

            delta = torch.cat(
                (br_c.unsqueeze(-1) * c_c.unsqueeze(-2), bi_c.unsqueeze(-1) * c_c.unsqueeze(-2)),
                dim=-1,
            )  # [B, M, L, F, 4F] = (real | imag)
            cum, n = _decayed_chunk_scan(
                delta, valid[:, start:end], w, carry, state, count
            )
            state = cum[:, :, -1]
            count = n[:, :, -1]

            inv_n = 1.0 / n.clamp_min(1e-6)[..., None, None]
            kr_c, ki_c = self._normalize_kernel(*(cum * inv_n).chunk(2, dim=-1), inv_n)

            o_c = (
                torch.matmul(ar_c.unsqueeze(-2), kr_c).squeeze(-2)
                + torch.matmul(ai_c.unsqueeze(-2), ki_c).squeeze(-2)
            ) * inv_sqrt_f
            outputs.append(o_c.to(dtype=c_cat.dtype))

        o_cat = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=2)
        return o_cat.chunk(2, dim=-1)

    def forward(
        self, x: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        _batch, _frames, bands, _dim = x.shape
        a, b, c, out_gate = self._project_abc(x)
        a, b, c, valid = self._mask_abc(a, b, c, frame_mask)

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
            o_0, o_1 = self._causal_forward(a_r, a_i, b_r, b_i, c_cat, valid)
            return self._finish_block(x, o_0, o_1, out_gate)

        inv_t = self._global_inv_count(valid)
        inv_sqrt_f = 1.0 / math.sqrt(bands)
        k_r = torch.matmul(b_r.transpose(-2, -1), c_cat) * inv_t.to(dtype=c_cat.dtype)
        k_i = torch.matmul(b_i.transpose(-2, -1), c_cat) * inv_t.to(dtype=c_cat.dtype)
        k_r, k_i = self._normalize_kernel(k_r, k_i, inv_t)
        o_cat = (torch.matmul(a_r, k_r) + torch.matmul(a_i, k_i)) * inv_sqrt_f
        o_0, o_1 = o_cat.chunk(2, dim=-1)
        return self._finish_block(x, o_0, o_1, out_gate)


class WindowedTimeTriangleRoPEBlock(TriangleRoPEBlock):
    """Local (optionally half-shifted) windowed time-to-time real-RoPE contraction.

    Frames are tiled into non-overlapping windows of `time_window` frames (after a
    left pad of `window_shift`). Each window w gets its own transfer matrix
        K_r^w[f', f] = (1/|w|) * sum_{t' in w} b_r[t', f'] * c[t', f]
        K_i^w[f', f] = (1/|w|) * sum_{t' in w} b_i[t', f'] * c[t', f]
    (jointly normalized as in `TimeTriangleRoPEBlock`, `|w|` = real frames) and
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
        soft_norm: bool = True,
        norm_floor_init: float = 0.5,
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
            soft_norm=soft_norm,
            norm_floor_init=norm_floor_init,
        )
        if time_window <= 0:
            raise ValueError(f"time_window must be positive, got {time_window}.")
        if not 0 <= window_shift < time_window:
            raise ValueError(
                f"window_shift ({window_shift}) must be in [0, time_window)."
            )
        self.time_window = time_window
        self.time_window_shift = window_shift

    def forward(
        self, x: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, frames, bands, _dim = x.shape
        a, b, c, out_gate = self._project_abc(x)
        a, b, c, valid = self._mask_abc(a, b, c, frame_mask)

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

        # Number of real (unpadded, unmasked) frames per window, for an unbiased mean.
        inv_count = self._window_inv_counts(valid, shift, pad_right, win)

        k_r = torch.matmul(b_rw.transpose(-2, -1), c_w) * inv_count.to(dtype=c_w.dtype)
        k_i = torch.matmul(b_iw.transpose(-2, -1), c_w) * inv_count.to(dtype=c_w.dtype)
        k_r, k_i = self._normalize_kernel(k_r, k_i, inv_count)

        inv_sqrt_f = 1.0 / math.sqrt(bands)
        o_w = (torch.matmul(a_rw, k_r) + torch.matmul(a_iw, k_i)) * inv_sqrt_f
        o_cat = o_w.reshape(batch, m_ch, padded_t, two_f)[:, :, shift : shift + frames, :]
        o_0, o_1 = o_cat.chunk(2, dim=-1)
        return self._finish_block(x, o_0, o_1, out_gate)


class GlobalFreqTriangleRoPEBlock(TriangleRoPEBlock):
    """Global chunk-wide real-RoPE band-to-band softmax affinity:
        M[f, f'] = Re(mean_{t'} (R_f a[t', f]) * conj(R_{f'} b[t', f']))
        A[f, :]  = softmax([norm(M[f, :]) * freq_scale, sink])[:F]
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
        soft_norm: bool = True,
        norm_floor_init: float = 0.5,
        attn_sink: bool = True,
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
            soft_norm=soft_norm,
            norm_floor_init=norm_floor_init,
            attn_sink=attn_sink,
        )

    def forward(
        self, x: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        a, b, c, out_gate = self._project_abc(x)
        a, b, c, valid = self._mask_abc(a, b, c, frame_mask)

        a_r, a_i, b_r, b_i = self._apply_freq_rope(a, b)
        c_0, c_1 = c.chunk(2, dim=-1)

        a_r = a_r.permute(0, 3, 1, 2)
        a_i = a_i.permute(0, 3, 1, 2)
        b_r = b_r.permute(0, 3, 1, 2)
        b_i = b_i.permute(0, 3, 1, 2)
        c_0 = c_0.permute(0, 3, 1, 2)
        c_1 = c_1.permute(0, 3, 1, 2)

        inv_t = self._global_inv_count(valid)
        m = (
            torch.matmul(a_r.transpose(-2, -1), b_r)
            + torch.matmul(a_i.transpose(-2, -1), b_i)
        ) * inv_t.to(dtype=a_r.dtype)
        attn = self._freq_attention(m, inv_t).to(dtype=c_0.dtype)
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
        soft_norm: bool = True,
        norm_floor_init: float = 0.5,
        attn_sink: bool = True,
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
            soft_norm=soft_norm,
            norm_floor_init=norm_floor_init,
            attn_sink=attn_sink,
        )

    def forward(
        self, x: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, frames, bands, _dim = x.shape
        a, b, c, out_gate = self._project_abc(x)
        a, b, c, valid = self._mask_abc(a, b, c, frame_mask)

        a_r, a_i, b_r, b_i = self._apply_freq_rope(a, b)
        c_0, c_1 = c.chunk(2, dim=-1)

        a_r = a_r.permute(0, 3, 1, 2)
        a_i = a_i.permute(0, 3, 1, 2)
        b_r = b_r.permute(0, 3, 1, 2)
        b_i = b_i.permute(0, 3, 1, 2)
        c_0 = c_0.permute(0, 3, 1, 2)
        c_1 = c_1.permute(0, 3, 1, 2)

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

        inv_count = self._window_inv_counts(valid, shift, pad_right, win)
        m = (
            torch.matmul(a_rw.transpose(-2, -1), b_rw)
            + torch.matmul(a_iw.transpose(-2, -1), b_iw)
        ) * inv_count.to(dtype=a_rw.dtype)
        attn = self._freq_attention(m, inv_count).to(dtype=c_0.dtype)

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


class CausalFreqTriangleRoPEBlock(TriangleRoPEBlock):
    """Causal band-to-band affinity pooled over an exponentially decayed prefix:
        M_t[f, f'] = sum_{t' <= t} g^(t - t') m_{t'}[f, f'] / sum_{t' <= t} g^(t - t')
        m_t[f, f'] = Re((R_f a[t, f]) * conj(R_{f'} b[t, f']))
        O[t, f]    = sum_{f'} softmax_{f'}(M_t[f, :])[f'] * c[t, f'],
    with a learnable per-channel decay `g` (half-lives initialized log-spaced over
    `decay_half_life` frames). Frame `t` only depends on frames `<= t`, but unlike
    a frame-local affinity (rank 2 per channel) it keeps co-modulation statistics
    across time. Computed in chunks of `chunk_size`; memory per chunk is
    `O(B * M * chunk_size * F^2)`, like the causal `TimeTriangleRoPEBlock`.
    `causal_decay=False` gives the plain running mean.
    """

    def __init__(
        self,
        dim: int,
        tri_dim: int | None = 128,
        heads: int | None = None,
        ffn_mult: float = 3.0,
        rope_base: float = 1_000.0,
        chunk_size: int = 32,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        freq_positions: torch.Tensor | None = None,
        soft_norm: bool = True,
        norm_floor_init: float = 0.5,
        attn_sink: bool = True,
        causal_decay: bool = True,
        decay_half_life: tuple[float, float] = (2.0, 512.0),
    ):
        super().__init__(
            dim=dim,
            tri_dim=tri_dim,
            heads=heads,
            axis="freq",
            ffn_mult=ffn_mult,
            rope_base=rope_base,
            dropout=dropout,
            residual_scale=residual_scale,
            freq_positions=freq_positions,
            soft_norm=soft_norm,
            norm_floor_init=norm_floor_init,
            attn_sink=attn_sink,
        )
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
        self.chunk_size = chunk_size
        if causal_decay:
            self.freq_decay_logit = nn.Parameter(
                _decay_logit_init(self.heads, *decay_half_life)
            )
        else:
            self.register_parameter("freq_decay_logit", None)

    def forward(
        self, x: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, frames, bands, _dim = x.shape
        a, b, c, out_gate = self._project_abc(x)
        a, b, c, valid = self._mask_abc(a, b, c, frame_mask)

        a_r, a_i, b_r, b_i = self._apply_freq_rope(a, b)
        c_0, c_1 = c.chunk(2, dim=-1)

        a_r = a_r.permute(0, 3, 1, 2)
        a_i = a_i.permute(0, 3, 1, 2)
        b_r = b_r.permute(0, 3, 1, 2)
        b_i = b_i.permute(0, 3, 1, 2)
        c_0 = c_0.permute(0, 3, 1, 2)
        c_1 = c_1.permute(0, 3, 1, 2)
        m_ch = a_r.shape[1]

        log_gamma = (
            None
            if self.freq_decay_logit is None
            else F.logsigmoid(self.freq_decay_logit.float())
        )
        state = torch.zeros(
            batch, m_ch, bands, bands, device=x.device, dtype=torch.float32
        )
        count = torch.zeros(batch, m_ch, device=x.device, dtype=torch.float32)
        outs_0: list[torch.Tensor] = []
        outs_1: list[torch.Tensor] = []

        for start in range(0, frames, self.chunk_size):
            end = min(start + self.chunk_size, frames)
            w, carry = _decay_weights(log_gamma, end - start, x.device)
            delta = torch.einsum(
                "bmlf,bmlg->bmlfg",
                a_r[:, :, start:end].float(),
                b_r[:, :, start:end].float(),
            ) + torch.einsum(
                "bmlf,bmlg->bmlfg",
                a_i[:, :, start:end].float(),
                b_i[:, :, start:end].float(),
            )
            cum, n = _decayed_chunk_scan(
                delta, valid[:, start:end], w, carry, state, count
            )
            state = cum[:, :, -1]
            count = n[:, :, -1]
            inv_n = 1.0 / n.clamp_min(1e-6)[..., None, None]
            attn = self._freq_attention(cum * inv_n, inv_n).to(dtype=c_0.dtype)
            outs_0.append(torch.einsum("bmlfg,bmlg->bmlf", attn, c_0[:, :, start:end]))
            outs_1.append(torch.einsum("bmlfg,bmlg->bmlf", attn, c_1[:, :, start:end]))

        o_0 = outs_0[0] if len(outs_0) == 1 else torch.cat(outs_0, dim=2)
        o_1 = outs_1[0] if len(outs_1) == 1 else torch.cat(outs_1, dim=2)
        return self._finish_block(x, o_0, o_1, out_gate)


def _resolve_attn_heads(dim: int, heads: int | None) -> int:
    resolved = _resolve_num_heads(dim, heads, target_head_dim=32)
    head_dim = dim // resolved
    if head_dim % 2 != 0:
        raise ValueError(
            f"RoPE attention requires an even head_dim; got dim={dim}, heads={resolved} "
            f"(head_dim={head_dim})."
        )
    return resolved


class RoPEAttentionBlock(nn.Module):
    """Base standard multi-head RoPE softmax attention block for `[B, T, F, D]` grids.

    Uses `F.scaled_dot_product_attention` with QK-RMSNorm and optional learnable
    per-head null/sink key (`attn_sink=True`, paired with a zero value token so
    queries can attend to "nothing" without breaking SDPA/FlashAttention).

    Subclasses pair this with their fixed FFN:
      - `TimeRoPEAttentionBlock`: SwiGLU FFN.
      - `FreqRoPEAttentionBlock`: `HopfieldFFN`.
    """

    def __init__(
        self,
        dim: int,
        heads: int | None = None,
        axis: str = "time",
        rope_base: float = 10_000.0,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        freq_positions: torch.Tensor | None = None,
        attn_sink: bool = True,
    ):
        super().__init__()
        if axis not in ("time", "freq"):
            raise ValueError(f"axis must be 'time' or 'freq', got {axis!r}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        resolved_heads = _resolve_attn_heads(dim, heads)
        self.dim = dim
        self.heads = resolved_heads
        self.head_dim = dim // resolved_heads
        self.axis = axis
        self.dropout = dropout

        self.attn_norm = nn.RMSNorm(dim)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)
        if axis == "freq":
            if freq_positions is None:
                raise ValueError(
                    "Frequency-axis RoPE attention blocks require freq_positions "
                    "(per-band log-frequency positions)."
                )
            freq_cos, freq_sin = self.rope.freq_tables(torch.as_tensor(freq_positions))
            self.register_buffer("freq_cos", freq_cos, persistent=False)
            self.register_buffer("freq_sin", freq_sin, persistent=False)
        else:
            self.register_buffer("freq_cos", None, persistent=False)
            self.register_buffer("freq_sin", None, persistent=False)

        if attn_sink:
            self.sink_k: nn.Parameter | None = nn.Parameter(
                torch.zeros(1, resolved_heads, 1, self.head_dim)
            )
        else:
            self.register_parameter("sink_k", None)

        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.RMSNorm(dim)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        in_std = 1.0 / math.sqrt(self.dim)
        nn.init.normal_(self.qkv_proj.weight, std=in_std)
        nn.init.normal_(self.out_proj.weight, std=in_std * residual_scale)

    def _prepend_sink(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sink_k is None:
            return k, v
        batch_n = k.shape[0]
        sk = self.sink_k.to(dtype=k.dtype).expand(batch_n, -1, -1, -1)
        sv = torch.zeros_like(sk)
        return torch.cat((sk, k), dim=-2), torch.cat((sv, v), dim=-2)


class TimeRoPEAttentionBlock(RoPEAttentionBlock):
    """Standard multi-head RoPE softmax attention along the time axis `T`
    (independent per frequency band `F`) with SwiGLU FFN, for sharp
    frame-to-frame recall alongside linear-time Triangle-RoPE contractions.
    """

    def __init__(
        self,
        dim: int,
        heads: int | None = None,
        causal: bool = False,
        ffn_mult: float = 3.0,
        rope_base: float = 10_000.0,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        attn_sink: bool = True,
    ):
        super().__init__(
            dim=dim,
            heads=heads,
            axis="time",
            rope_base=rope_base,
            dropout=dropout,
            residual_scale=residual_scale,
            attn_sink=attn_sink,
        )
        self.causal = causal
        ffn_hidden = max(64, int(round(dim * ffn_mult / 64.0)) * 64)
        self.ffn_in = nn.Linear(dim, ffn_hidden * 2, bias=False)
        self.ffn_out = nn.Linear(ffn_hidden, dim, bias=False)
        nn.init.normal_(self.ffn_in.weight, std=1.0 / math.sqrt(dim))
        nn.init.normal_(
            self.ffn_out.weight,
            std=(1.0 / math.sqrt(ffn_hidden)) * residual_scale,
        )

    def _sdpa_mask(
        self,
        batch: int,
        bands: int,
        frames: int,
        device: torch.device,
        frame_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, bool]:
        has_sink = self.sink_k is not None
        if frame_mask is None:
            if not self.causal:
                return None, False
            if not has_sink:
                return None, True
            tril = torch.tril(torch.ones(frames, frames, dtype=torch.bool, device=device))
            sink_col = torch.ones(frames, 1, dtype=torch.bool, device=device)
            mask = torch.cat((sink_col, tril), dim=-1)[None, None, :, :]
            return mask, False

        valid = frame_mask.to(device=device) > 0.5  # [B, T]
        if self.causal:
            tril = torch.tril(torch.ones(frames, frames, dtype=torch.bool, device=device))
            m = tril[None, :, :] & valid[:, None, :]  # [B, T, T]
            if has_sink:
                sink_col = torch.ones(batch, frames, 1, dtype=torch.bool, device=device)
                m = torch.cat((sink_col, m), dim=-1)  # [B, T, T + 1]
            else:
                m = m | (~m.any(dim=-1, keepdim=True))
            m = m[:, None, None, :, :].expand(batch, bands, 1, frames, m.shape[-1])
            return m.reshape(batch * bands, 1, frames, m.shape[-1]), False

        if has_sink:
            sink_col = torch.ones(batch, 1, dtype=torch.bool, device=device)
            k_valid = torch.cat((sink_col, valid), dim=-1)  # [B, T + 1]
        else:
            k_valid = valid | (~valid.any(dim=-1, keepdim=True))
        k_valid = k_valid[:, None, None, None, :].expand(
            batch, bands, 1, 1, k_valid.shape[-1]
        )
        return k_valid.reshape(batch * bands, 1, 1, k_valid.shape[-1]), False

    def forward(
        self, x: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, frames, bands, dim = x.shape
        h = self.attn_norm(x).permute(0, 2, 1, 3).reshape(batch * bands, frames, dim)
        q, k, v = self.qkv_proj(h).chunk(3, dim=-1)
        q = self.q_norm(
            q.reshape(batch * bands, frames, self.heads, self.head_dim).transpose(1, 2)
        )
        k = self.k_norm(
            k.reshape(batch * bands, frames, self.heads, self.head_dim).transpose(1, 2)
        )
        v = v.reshape(batch * bands, frames, self.heads, self.head_dim).transpose(1, 2)

        q, k = self.rope.apply_qk(q, k)
        k, v = self._prepend_sink(k, v)
        attn_mask, is_causal = self._sdpa_mask(batch, bands, frames, x.device, frame_mask)
        dropout_p = self.dropout if self.training else 0.0
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )
        o = (
            o.transpose(1, 2)
            .contiguous()
            .reshape(batch, bands, frames, dim)
            .permute(0, 2, 1, 3)
        )
        x = x + self.drop(self.out_proj(o))
        ffn_gate, ffn_val = self.ffn_in(self.ffn_norm(x)).chunk(2, dim=-1)
        ffn = F.silu(ffn_gate) * ffn_val
        return x + self.drop(self.ffn_out(self.drop(ffn)))


class FreqRoPEAttentionBlock(RoPEAttentionBlock):
    """Per-frame multi-head RoPE softmax attention along the frequency axis `F`
    (using interleaved log-frequency + band-index RoPE tables) paired with
    `HopfieldFFN` for instantaneous harmonic routing + timbral prototype retrieval.
    """

    def __init__(
        self,
        dim: int,
        heads: int | None = None,
        rope_base: float = 1_000.0,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        freq_positions: torch.Tensor | None = None,
        attn_sink: bool = True,
        hopfield_heads: int | None = None,
        hopfield_patterns: int = 256,
        hopfield_steps: int = 2,
        hopfield_step_size: float = 1.0,
    ):
        super().__init__(
            dim=dim,
            heads=heads,
            axis="freq",
            rope_base=rope_base,
            dropout=dropout,
            residual_scale=residual_scale,
            freq_positions=freq_positions,
            attn_sink=attn_sink,
        )
        self.hopfield_ffn = HopfieldFFN(
            dim=dim,
            num_heads=hopfield_heads,
            num_patterns=hopfield_patterns,
            num_steps=hopfield_steps,
            step_size=hopfield_step_size,
            dropout=dropout,
            residual_scale=residual_scale,
            attn_sink=attn_sink,
        )

    def forward(
        self, x: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        del frame_mask  # strictly per-frame; padded frames cannot affect real frames
        batch, frames, bands, dim = x.shape
        h = self.attn_norm(x).reshape(batch * frames, bands, dim)
        q, k, v = self.qkv_proj(h).chunk(3, dim=-1)
        q = self.q_norm(
            q.reshape(batch * frames, bands, self.heads, self.head_dim).transpose(1, 2)
        )
        k = self.k_norm(
            k.reshape(batch * frames, bands, self.heads, self.head_dim).transpose(1, 2)
        )
        v = v.reshape(batch * frames, bands, self.heads, self.head_dim).transpose(1, 2)

        assert self.freq_cos is not None and self.freq_sin is not None
        q, k = self.rope.apply_qk_tables(q, k, self.freq_cos, self.freq_sin)
        k, v = self._prepend_sink(k, v)
        dropout_p = self.dropout if self.training else 0.0
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        o = o.transpose(1, 2).contiguous().reshape(batch, frames, bands, dim)
        x = x + self.drop(self.out_proj(o))
        return x + self.hopfield_ffn(self.ffn_norm(x))


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
    causal: bool = False,
    chunk_size: int = 32,
    soft_norm: bool = True,
    norm_floor_init: float = 0.5,
    attn_sink: bool = True,
    causal_decay: bool = True,
) -> TriangleRoPEBlock:
    """`causal=True` builds a decayed-prefix affinity (window args ignored);
    otherwise `freq_window > 0` builds a windowed affinity and `freq_window == 0`
    a chunk-wide one."""
    common = dict(
        dim=dim,
        tri_dim=tri_dim,
        heads=heads,
        ffn_mult=ffn_mult,
        rope_base=rope_base,
        dropout=dropout,
        residual_scale=residual_scale,
        freq_positions=freq_positions,
        soft_norm=soft_norm,
        norm_floor_init=norm_floor_init,
        attn_sink=attn_sink,
    )
    if causal:
        return CausalFreqTriangleRoPEBlock(
            chunk_size=chunk_size, causal_decay=causal_decay, **common
        )
    if freq_window > 0:
        return WindowedFreqTriangleRoPEBlock(
            freq_window=freq_window, window_shift=window_shift, **common
        )
    return GlobalFreqTriangleRoPEBlock(**common)


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
    soft_norm: bool = True,
    norm_floor_init: float = 0.5,
    causal_decay: bool = True,
) -> TriangleRoPEBlock:
    """`time_window > 0` builds a local windowed (non-causal) time contraction;
    `time_window == 0` builds the chunk-wide (or causal prefix) contraction."""
    common = dict(
        dim=dim,
        tri_dim=tri_dim,
        heads=heads,
        ffn_mult=ffn_mult,
        rope_base=rope_base,
        dropout=dropout,
        residual_scale=residual_scale,
        soft_norm=soft_norm,
        norm_floor_init=norm_floor_init,
    )
    if time_window > 0:
        if causal:
            raise ValueError("Windowed time contraction is non-causal; use time_window=0.")
        return WindowedTimeTriangleRoPEBlock(
            time_window=time_window, window_shift=window_shift, **common
        )
    return TimeTriangleRoPEBlock(
        causal=causal, chunk_size=chunk_size, causal_decay=causal_decay, **common
    )
