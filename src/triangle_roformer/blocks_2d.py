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


def _causal_time_weights(
    log_gamma: torch.Tensor | None,
    length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intra-chunk decay matrices `(w, w_sq, w_strict, carry, carry_sq)` for the
    O(L^2 * F + L * F^2) algebraic causal time contraction."""
    idx = torch.arange(length, device=device, dtype=torch.float32)
    lag = idx[:, None] - idx[None, :]
    lower = lag >= 0
    strict = lag > 0
    if log_gamma is None:
        w = lower.float()[None]
        carry = torch.ones(1, length, device=device)
        return w, w, strict.float()[None], carry, carry
    lg = log_gamma.float().to(device)
    lag_pos = lag.clamp_min(0.0)[None]
    w = torch.where(lower[None], torch.exp(lg.view(-1, 1, 1) * lag_pos), 0.0)
    w_sq = torch.where(lower[None], torch.exp((2.0 * lg).view(-1, 1, 1) * lag_pos), 0.0)
    w_strict = torch.where(strict[None], w, 0.0)
    step = (idx + 1.0)[None]
    carry = torch.exp(lg.view(-1, 1) * step)
    carry_sq = torch.exp((2.0 * lg).view(-1, 1) * step)
    return w, w_sq, w_strict, carry, carry_sq


def _decayed_chunk_scan(
    delta: torch.Tensor,
    valid: torch.Tensor,
    w: torch.Tensor,
    carry: torch.Tensor,
    state: torch.Tensor | None,
    count: torch.Tensor | None,
    use_cumsum: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decayed causal prefix sums over one chunk.

    delta: `[B, M, L, *rest]` per-frame terms (fp32), valid: `[B, L]` frame weights,
    state: optional `[B, M, *rest]` carried sum, count: optional `[B, M']` carried weight.
    Returns `cum [B, M, L, *rest]` and `n [B, M', L]` (decayed valid-frame counts).
    """
    batch, m_ch, length = delta.shape[:3]
    rest = delta.shape[3:]
    if use_cumsum:
        cum = delta.cumsum(dim=2)
        n = valid.cumsum(dim=1)[:, None, :]
        if state is not None:
            cum = cum + state[:, :, None]
        if count is not None:
            n = n + count[:, :, None]
        return cum, n
    flat = delta.reshape(batch, m_ch, length, -1)
    cum = torch.matmul(w, flat)
    n = torch.matmul(w, valid[:, None, :, None]).squeeze(-1)
    if state is not None:
        cum = cum + carry[None, :, :, None] * state.reshape(batch, m_ch, 1, -1)
    if count is not None:
        n = n + carry[None] * count[:, :, None]
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
        # Skip post-readout RMSNorm on freq blocks with attn_sink so all-head
        # sink suppression isn't rescaled back to unit RMS (c is already c_norm'd).
        self.tri_out_norm: nn.Module = (
            nn.Identity() if (axis == "freq" and attn_sink) else nn.RMSNorm(resolved_tri_dim)
        )
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
    def _mask_abc_opt(
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        frame_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Zero (a, b, c) on padded frames when `frame_mask` is provided; avoids
        allocating an all-ones mask when `frame_mask is None`."""
        if frame_mask is None:
            return a, b, c, None
        valid = frame_mask.to(device=a.device, dtype=torch.float32)
        m = valid.to(dtype=a.dtype)[:, :, None, None]
        return a * m, b * m, c * m, valid

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
        self, ndim: int, inv_count: torch.Tensor | float
    ) -> torch.Tensor | float:
        """Squared normalization floor `floor^2 / n` (or the plain eps without
        `soft_norm`). `inv_count` (fp32 tensor or float, `1 / n`) must broadcast."""
        if self.log_norm_floor is None:
            return 1e-6
        floor = self.log_norm_floor.float().exp().clamp(_FLOOR_MIN, _FLOOR_MAX)
        return self._channel_view(floor.square(), ndim) * inv_count + 1e-6

    def _normalize_kernel(
        self, k_r: torch.Tensor, k_i: torch.Tensor, inv_count: torch.Tensor | float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Phase-preserving joint normalization of `(K_r, K_i)` over source bands
        (dim -2). With `soft_norm`, divides by `sqrt(ms + floor^2 / n)` so the
        kernel keeps its correlation strength when it is near the noise level."""
        ms = (k_r.float().square() + k_i.float().square()).mean(dim=-2, keepdim=True)
        scale = torch.rsqrt(ms + self._floor_sq(ms.ndim, inv_count))
        return k_r * scale.to(dtype=k_r.dtype), k_i * scale.to(dtype=k_i.dtype)

    def _normalize_kernel_packed(
        self, k_ri: torch.Tensor, bands: int, inv_count: torch.Tensor | float
    ) -> torch.Tensor:
        """Phase-preserving joint normalization of packed `[..., 2*F, 2*F]` kernel."""
        ms = (
            k_ri.float()
            .square()
            .reshape(*k_ri.shape[:-2], 2, bands, k_ri.shape[-1])
            .sum(dim=-3)
            .mean(dim=-2, keepdim=True)
        )
        scale = torch.rsqrt(ms + self._floor_sq(ms.ndim, inv_count))
        return k_ri * scale.to(dtype=k_ri.dtype)

    def _sink_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        if self.sink_logit is None:
            return F.softmax(logits, dim=-1)
        sink = self._channel_view(self.sink_logit, logits.ndim)
        if torch.compiler.is_compiling():
            m_max = torch.maximum(logits.amax(dim=-1, keepdim=True), sink)
            exp_logits = (logits - m_max).exp()
            denom = exp_logits.sum(dim=-1, keepdim=True) + (sink - m_max).exp()
            return exp_logits / denom
        return F.softmax(
            torch.cat((logits, sink.expand(*logits.shape[:-1], 1)), dim=-1), dim=-1
        )[..., :-1]

    def _freq_attention(
        self, m: torch.Tensor, inv_count: torch.Tensor | float
    ) -> torch.Tensor:
        """Softmax over the last (source band) axis of a `[B, M, ..., F, F]`
        affinity, with optional soft normalization and a null (sink) logit."""
        assert self.freq_scale is not None
        m_f = m.float()
        ms = m_f.square().mean(dim=-1, keepdim=True)
        scale = self._channel_view(self.freq_scale.clamp(0.1, 20.0), m_f.ndim)
        logits = m_f * torch.rsqrt(ms + self._floor_sq(m_f.ndim, inv_count)) * scale
        return self._sink_softmax(logits)

    def _freq_attention_sum(
        self, cum: torch.Tensor, count: torch.Tensor
    ) -> torch.Tensor:
        """Like `_freq_attention(cum / count, 1 / count)` without materializing
        an intermediate `cum / count` copy."""
        assert self.freq_scale is not None
        cum_f = cum.float()
        n_safe = count.clamp_min(1e-6)[..., None, None]
        ms = cum_f.square().mean(dim=-1, keepdim=True)
        if self.log_norm_floor is None:
            floor_term = 1e-6 * n_safe.square()
        else:
            floor = self.log_norm_floor.float().exp().clamp(_FLOOR_MIN, _FLOOR_MAX)
            floor_term = (
                self._channel_view(floor.square(), cum_f.ndim) * n_safe
                + 1e-6 * n_safe.square()
            )
        scale = self._channel_view(self.freq_scale.clamp(0.1, 20.0), cum_f.ndim)
        logits = cum_f * torch.rsqrt(ms + floor_term) * scale
        return self._sink_softmax(logits)

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

    def _finish_block_packed(
        self,
        x: torch.Tensor,
        tri: torch.Tensor,
        out_gate: torch.Tensor,
    ) -> torch.Tensor:
        tri = self.tri_out(self.tri_out_norm(tri)) * torch.sigmoid(out_gate)
        x = x + self.drop(tri)

        h = self.ffn_norm(x)
        ffn_gate, ffn_val = self.ffn_in(h).chunk(2, dim=-1)
        ffn = F.silu(ffn_gate) * ffn_val
        return x + self.drop(self.ffn_out(self.drop(ffn)))

    def _finish_block(
        self,
        x: torch.Tensor,
        o_0: torch.Tensor,
        o_1: torch.Tensor,
        out_gate: torch.Tensor,
    ) -> torch.Tensor:
        tri = torch.cat((o_0, o_1), dim=1).permute(0, 2, 3, 1).contiguous()
        return self._finish_block_packed(x, tri, out_gate)


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
        a_ri: torch.Tensor,
        b_ri: torch.Tensor,
        c_cat: torch.Tensor,
        valid: torch.Tensor | None,
        bands: int,
    ) -> torch.Tensor:
        """Exact algebraic causal prefix contraction in O(B*M*T*(F^2 + L*F)) without
        materializing 5D `[B, M, L, 2F, 2F]` chunk states."""
        batch, m_ch, frames, _two_f = a_ri.shape
        inv_sqrt_f = 1.0 / math.sqrt(bands)
        inv_f = 1.0 / bands
        log_gamma = (
            None
            if self.time_decay_logit is None
            else F.logsigmoid(self.time_decay_logit.float())
        )
        use_cumsum = log_gamma is None

        state: torch.Tensor | None = None
        count: torch.Tensor | None = None
        outputs: list[torch.Tensor] = []

        full_len = min(self.chunk_size, frames)
        full_weights = _causal_time_weights(log_gamma, full_len, a_ri.device)

        for start in range(0, frames, self.chunk_size):
            end = min(start + self.chunk_size, frames)
            chunk_len = end - start
            w, w_sq, w_strict, carry, carry_sq = (
                full_weights
                if chunk_len == full_len
                else _causal_time_weights(log_gamma, chunk_len, a_ri.device)
            )

            a_c = a_ri[:, :, start:end].float()  # [B, M, L, 2F]
            b_c = b_ri[:, :, start:end].float()  # [B, M, L, 2F]
            c_c = c_cat[:, :, start:end].float()  # [B, M, L, 2F]

            if valid is None:
                n = w.sum(dim=-1)[None]  # [1, M', L]
            elif use_cumsum:
                n = valid[:, start:end].cumsum(dim=1)[:, None, :]  # [B, 1, L]
            else:
                n = torch.matmul(w, valid[:, None, start:end, None]).squeeze(-1)  # [B, M, L]
            if count is not None:
                n = n + carry[None] * count[:, :, None]

            # Numerator U = carry * (a_c @ state) + ((a_c @ b_c^T) * w) @ c_c
            ab_w = torch.matmul(a_c, b_c.transpose(-2, -1)) * w[None]  # [B, M, L, L]
            u_c = torch.matmul(ab_w, c_c)

            # Intra-chunk norm squared: ||sum_{i<=t} w[t,i] b_i c_i||^2
            gram = torch.matmul(b_c, b_c.transpose(-2, -1))  # [B, M, L, L]
            d_strict = torch.matmul(gram * w_strict[None], c_c)  # [B, M, L, 2F]
            g_diag = gram.diagonal(dim1=-2, dim2=-1)[..., None]  # [B, M, L, 1]
            dq = g_diag * c_c.square() + 2.0 * c_c * d_strict
            sum_sq = dq.cumsum(dim=2) if use_cumsum else torch.matmul(w_sq[None], dq)

            if state is not None:
                u_c = u_c + carry[None, :, :, None] * torch.matmul(a_c, state)
                state_sq = state.square().sum(dim=-2, keepdim=True)  # [B, M, 1, 2F]
                bs_c = torch.matmul(b_c, state) * c_c
                cross_scan = (
                    bs_c.cumsum(dim=2) if use_cumsum else torch.matmul(w[None], bs_c)
                )
                sum_sq = (
                    sum_sq
                    + carry_sq[None, :, :, None] * state_sq
                    + (2.0 * carry[None, :, :, None]) * cross_scan
                )

            inv_n = 1.0 / n.clamp_min(1e-6)[..., None]  # [B or 1, M', L, 1]
            ms = sum_sq.clamp_min(0.0) * (inv_n.square() * inv_f)
            scale = torch.rsqrt(ms + self._floor_sq(ms.ndim, inv_n))
            o_c = (u_c * inv_n) * scale * inv_sqrt_f
            outputs.append(o_c.to(dtype=c_cat.dtype))

            if end < frames:
                b_weighted = b_c * w[None, :, -1, :, None]
                chunk_state = torch.matmul(b_weighted.transpose(-2, -1), c_c)
                if state is None:
                    state = chunk_state
                else:
                    state = carry[:, -1].view(1, -1, 1, 1) * state + chunk_state
                count = n[:, :, -1]

        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=2)

    def forward(
        self, x: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, frames, bands, _dim = x.shape
        a, b, c, out_gate = self._project_abc(x)
        a, b, c, valid = self._mask_abc_opt(a, b, c, frame_mask)

        a_r, a_i, b_r, b_i = self.rope.apply_time(a, b)
        m_ch = self.heads
        a_ri = torch.cat((a_r, a_i), dim=2).permute(0, 3, 1, 2)  # [B, M, T, 2F]
        b_ri = torch.cat((b_r, b_i), dim=2).permute(0, 3, 1, 2)  # [B, M, T, 2F]
        c_cat = (
            c.reshape(batch, frames, bands, 2, m_ch)
            .permute(0, 4, 1, 3, 2)
            .reshape(batch, m_ch, frames, 2 * bands)
        )  # [B, M, T, 2F]

        if self.causal:
            o_cat = self._causal_forward(a_ri, b_ri, c_cat, valid, bands)
        else:
            inv_t: torch.Tensor | float = (
                1.0 / frames if valid is None else self._global_inv_count(valid)
            )
            inv_t_val = (
                inv_t if isinstance(inv_t, float) else inv_t.to(dtype=c_cat.dtype)
            )
            k_ri = torch.matmul(b_ri.transpose(-2, -1), c_cat) * inv_t_val
            k_ri = self._normalize_kernel_packed(k_ri, bands, inv_t)
            o_cat = torch.matmul(a_ri, k_ri) * (1.0 / math.sqrt(bands))

        tri = (
            o_cat.reshape(batch, m_ch, frames, 2, bands)
            .permute(0, 2, 4, 3, 1)
            .reshape(batch, frames, bands, 2 * m_ch)
        )
        return self._finish_block_packed(x, tri, out_gate)


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
        a, b, c, valid = self._mask_abc_opt(a, b, c, frame_mask)

        a_r, a_i, b_r, b_i = self.rope.apply_time(a, b)
        m_ch = self.heads
        two_f = 2 * bands
        a_ri = torch.cat((a_r, a_i), dim=2).permute(0, 3, 1, 2)  # [B, M, T, 2F]
        b_ri = torch.cat((b_r, b_i), dim=2).permute(0, 3, 1, 2)  # [B, M, T, 2F]
        c_cat = (
            c.reshape(batch, frames, bands, 2, m_ch)
            .permute(0, 4, 1, 3, 2)
            .reshape(batch, m_ch, frames, two_f)
        )

        win = self.time_window
        shift = self.time_window_shift
        pad_right = (-(frames + shift)) % win
        needs_pad = shift > 0 or pad_right > 0
        if needs_pad:
            pad_spec = (0, 0, shift, pad_right)
            a_ri = F.pad(a_ri, pad_spec)
            b_ri = F.pad(b_ri, pad_spec)
            c_cat = F.pad(c_cat, pad_spec)

        padded_t = a_ri.shape[2]
        num_win = padded_t // win

        a_riw = a_ri.reshape(batch, m_ch, num_win, win, two_f)
        b_riw = b_ri.reshape(batch, m_ch, num_win, win, two_f)
        c_w = c_cat.reshape(batch, m_ch, num_win, win, two_f)

        # Number of real (unpadded, unmasked) frames per window, for an unbiased mean.
        if valid is None:
            if needs_pad:
                valid_1 = torch.ones(1, frames, device=x.device)
                inv_count: torch.Tensor | float = self._window_inv_counts(
                    valid_1, shift, pad_right, win
                )
            else:
                inv_count = 1.0 / win
        else:
            inv_count = self._window_inv_counts(valid, shift, pad_right, win)

        inv_val = (
            inv_count
            if isinstance(inv_count, float)
            else inv_count.to(dtype=c_w.dtype)
        )
        k_ri = torch.matmul(b_riw.transpose(-2, -1), c_w) * inv_val
        k_ri = self._normalize_kernel_packed(k_ri, bands, inv_count)

        o_w = torch.matmul(a_riw, k_ri) * (1.0 / math.sqrt(bands))
        o_cat = o_w.reshape(batch, m_ch, padded_t, two_f)
        if needs_pad:
            o_cat = o_cat[:, :, shift : shift + frames, :]
        tri = (
            o_cat.reshape(batch, m_ch, frames, 2, bands)
            .permute(0, 2, 4, 3, 1)
            .reshape(batch, frames, bands, 2 * m_ch)
        )
        return self._finish_block_packed(x, tri, out_gate)


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
        batch, frames, bands, _dim = x.shape
        a, b, c, out_gate = self._project_abc(x)
        a, b, c, valid = self._mask_abc_opt(a, b, c, frame_mask)

        a_r, a_i, b_r, b_i = self._apply_freq_rope(a, b)
        m_ch = self.heads
        a_ri = torch.cat((a_r, a_i), dim=1).permute(0, 3, 1, 2)  # [B, M, 2T, F]
        b_ri = torch.cat((b_r, b_i), dim=1).permute(0, 3, 1, 2)  # [B, M, 2T, F]

        inv_t: torch.Tensor | float = (
            1.0 / frames if valid is None else self._global_inv_count(valid)
        )
        inv_val = inv_t if isinstance(inv_t, float) else inv_t.to(dtype=a_ri.dtype)
        m = torch.matmul(a_ri.transpose(-2, -1), b_ri) * inv_val
        attn = self._freq_attention(m, inv_t).to(dtype=c.dtype)

        c_freq = (
            c.reshape(batch, frames, bands, 2, m_ch)
            .permute(0, 4, 3, 1, 2)
            .reshape(batch, m_ch, 2 * frames, bands)
        )
        o_freq = torch.matmul(c_freq, attn.transpose(-2, -1))
        tri = (
            o_freq.reshape(batch, m_ch, 2, frames, bands)
            .permute(0, 3, 4, 2, 1)
            .reshape(batch, frames, bands, 2 * m_ch)
        )
        return self._finish_block_packed(x, tri, out_gate)


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
        a, b, c, valid = self._mask_abc_opt(a, b, c, frame_mask)

        a_r, a_i, b_r, b_i = self._apply_freq_rope(a, b)
        m_ch = self.heads
        win = self.freq_window
        shift = self.window_shift
        pad_right = (-(frames + shift)) % win
        needs_pad = shift > 0 or pad_right > 0
        if needs_pad:
            pad_spec = (0, 0, 0, 0, shift, pad_right)
            a_r = F.pad(a_r, pad_spec)
            a_i = F.pad(a_i, pad_spec)
            b_r = F.pad(b_r, pad_spec)
            b_i = F.pad(b_i, pad_spec)
            c = F.pad(c, pad_spec)

        padded_t = a_r.shape[1]
        num_win = padded_t // win

        a_riw = (
            torch.stack(
                (
                    a_r.reshape(batch, num_win, win, bands, m_ch),
                    a_i.reshape(batch, num_win, win, bands, m_ch),
                ),
                dim=2,
            )
            .permute(0, 5, 1, 2, 3, 4)
            .reshape(batch, m_ch, num_win, 2 * win, bands)
        )
        b_riw = (
            torch.stack(
                (
                    b_r.reshape(batch, num_win, win, bands, m_ch),
                    b_i.reshape(batch, num_win, win, bands, m_ch),
                ),
                dim=2,
            )
            .permute(0, 5, 1, 2, 3, 4)
            .reshape(batch, m_ch, num_win, 2 * win, bands)
        )

        if valid is None:
            if needs_pad:
                valid_1 = torch.ones(1, frames, device=x.device)
                inv_count: torch.Tensor | float = self._window_inv_counts(
                    valid_1, shift, pad_right, win
                )
            else:
                inv_count = 1.0 / win
        else:
            inv_count = self._window_inv_counts(valid, shift, pad_right, win)

        inv_val = (
            inv_count
            if isinstance(inv_count, float)
            else inv_count.to(dtype=a_riw.dtype)
        )
        m = torch.matmul(a_riw.transpose(-2, -1), b_riw) * inv_val
        attn = self._freq_attention(m, inv_count).to(dtype=c.dtype)

        c_w = (
            c.reshape(batch, num_win, win, bands, 2, m_ch)
            .permute(0, 5, 1, 4, 2, 3)
            .reshape(batch, m_ch, num_win, 2 * win, bands)
        )
        o_w = torch.matmul(c_w, attn.transpose(-2, -1))
        tri = (
            o_w.reshape(batch, m_ch, num_win, 2, win, bands)
            .permute(0, 2, 4, 5, 3, 1)
            .reshape(batch, padded_t, bands, 2 * m_ch)
        )
        if needs_pad:
            tri = tri[:, shift : shift + frames, :, :]
        return self._finish_block_packed(x, tri, out_gate)


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
        m_ch = self.heads
        a_ri = torch.stack((a_r, a_i), dim=-1).permute(0, 3, 1, 2, 4)  # [B, M, T, F, 2]
        b_ri = torch.stack((b_r, b_i), dim=-1).permute(0, 3, 1, 4, 2)  # [B, M, T, 2, F]
        c_01 = c.reshape(batch, frames, bands, 2, m_ch).permute(
            0, 4, 1, 2, 3
        )  # [B, M, T, F, 2]

        log_gamma = (
            None
            if self.freq_decay_logit is None
            else F.logsigmoid(self.freq_decay_logit.float())
        )
        use_cumsum = log_gamma is None
        state: torch.Tensor | None = None
        count: torch.Tensor | None = None
        outs: list[torch.Tensor] = []

        full_len = min(self.chunk_size, frames)
        full_wc = _decay_weights(log_gamma, full_len, x.device)

        for start in range(0, frames, self.chunk_size):
            end = min(start + self.chunk_size, frames)
            chunk_len = end - start
            w, carry = (
                full_wc
                if chunk_len == full_len
                else _decay_weights(log_gamma, chunk_len, x.device)
            )
            delta = torch.matmul(
                a_ri[:, :, start:end].float(), b_ri[:, :, start:end].float()
            )
            cum, n = _decayed_chunk_scan(
                delta, valid[:, start:end], w, carry, state, count, use_cumsum=use_cumsum
            )
            state = cum[:, :, -1]
            count = n[:, :, -1]
            attn = self._freq_attention_sum(cum, n).to(dtype=c.dtype)
            outs.append(torch.matmul(attn, c_01[:, :, start:end]))

        o_01 = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
        tri = o_01.permute(0, 2, 3, 4, 1).reshape(batch, frames, bands, 2 * m_ch)
        return self._finish_block_packed(x, tri, out_gate)


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
    per-head position-invariant null/sink logit (`attn_sink=True`, implemented via
    a padded coordinate with a zero value token so queries can attend to "nothing"
    without position-dependent RoPE oscillation or breaking SDPA/FlashAttention).

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
            self.sink_logit: nn.Parameter | None = nn.Parameter(
                torch.zeros(resolved_heads)
            )
        else:
            self.register_parameter("sink_logit", None)

        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.RMSNorm(dim)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        in_std = 1.0 / math.sqrt(self.dim)
        nn.init.normal_(self.qkv_proj.weight, std=in_std)
        nn.init.normal_(self.out_proj.weight, std=in_std * residual_scale)

    def _prepare_sdpa_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Prepare `(q, k, v, scale)` for fused SDPA. When `attn_sink=True`,
        pads `head_dim` to a multiple of 8 so the prepended sink token carries an
        exact position-invariant scalar `sink_logit` (unaffected by RoPE on `q`)."""
        scale = 1.0 / math.sqrt(self.head_dim)
        if self.sink_logit is None:
            return q, k, v, scale
        pad_d = 8 - (self.head_dim % 8)
        d_ext = self.head_dim + pad_d
        q_ext = F.pad(q, (0, pad_d), value=1.0)
        k_real = F.pad(k, (0, pad_d))
        sink_val = (
            (self.sink_logit.float() * math.sqrt(self.head_dim))
            .to(dtype=k.dtype)
            .view(1, self.heads, 1, 1)
        )
        k_sink = torch.cat(
            (
                torch.zeros(1, self.heads, 1, d_ext - 1, device=k.device, dtype=k.dtype),
                sink_val,
            ),
            dim=-1,
        ).expand(k.shape[0], -1, -1, -1)
        k_all = torch.cat((k_sink, k_real), dim=-2)
        v_ext = F.pad(v, (0, pad_d, 1, 0))
        return q_ext, k_all, v_ext, scale


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
        has_sink = self.sink_logit is not None
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
            q.reshape(batch * bands, frames, self.heads, self.head_dim)
        ).transpose(1, 2)
        k = self.k_norm(
            k.reshape(batch * bands, frames, self.heads, self.head_dim)
        ).transpose(1, 2)
        v = v.reshape(batch * bands, frames, self.heads, self.head_dim).transpose(1, 2)

        q, k = self.rope.apply_qk(q, k)
        q, k, v, scale = self._prepare_sdpa_qkv(q, k, v)
        attn_mask, is_causal = self._sdpa_mask(batch, bands, frames, x.device, frame_mask)
        dropout_p = self.dropout if self.training else 0.0
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )
        if self.sink_logit is not None:
            o = o[..., : self.head_dim]
        o = self.out_proj(
            o.transpose(1, 2).reshape(batch, bands, frames, dim)
        ).permute(0, 2, 1, 3)
        x = x + self.drop(o)
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
            q.reshape(batch * frames, bands, self.heads, self.head_dim)
        ).transpose(1, 2)
        k = self.k_norm(
            k.reshape(batch * frames, bands, self.heads, self.head_dim)
        ).transpose(1, 2)
        v = v.reshape(batch * frames, bands, self.heads, self.head_dim).transpose(1, 2)

        assert self.freq_cos is not None and self.freq_sin is not None
        q, k = self.rope.apply_qk_tables(q, k, self.freq_cos, self.freq_sin)
        q, k, v, scale = self._prepare_sdpa_qkv(q, k, v)
        dropout_p = self.dropout if self.training else 0.0
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=scale)
        if self.sink_logit is not None:
            o = o[..., : self.head_dim]
        o = o.transpose(1, 2).reshape(batch, frames, bands, dim)
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
