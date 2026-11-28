from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


def log_frequency_positions(
    band_freqs_hz: Sequence[float],
    min_hz: float = 20.0,
) -> torch.Tensor:
    """Map band center frequencies (Hz) to RoPE positions in semitone units.

    `pos[f] = 12 * log2(max(hz[f], min_hz) / min_hz)`, so any fixed frequency
    *ratio* (octave = 12, fifth ~ 7.02, ...) is a fixed positional offset,
    independent of absolute pitch. Relative-RoPE contractions over these
    positions therefore see harmonic intervals identically for every F0.
    """
    if len(band_freqs_hz) == 0:
        raise ValueError("band_freqs_hz must contain at least one band.")
    if min_hz <= 0.0:
        raise ValueError(f"min_hz must be positive, got {min_hz}.")
    values = [float(hz) for hz in band_freqs_hz]
    if any(not math.isfinite(hz) or hz < 0.0 for hz in values):
        raise ValueError("band_freqs_hz must be finite and non-negative.")
    hz = torch.tensor(values, dtype=torch.float64).clamp_min(min_hz)
    return (12.0 * torch.log2(hz / min_hz)).to(torch.float32)


class RotaryEmbedding(nn.Module):
    """Complex Rotary Position Embedding (RoPE) supporting 1D sequences and 2D grids.

    Rotates paired real/imaginary halves of `(a, b)` along the requested axis:
    - `"1d"`: sequence axis of `[B, H, T, head_dim]` (dim 2) or `[B, T, dim]` (dim 1)
    - `"time"`: time axis of 2D `[B, T, F, tri_dim]` (dim 1)
    - `"freq"`: frequency axis of 2D `[B, T, F, tri_dim]` (dim 2)

    Supports an optional `offset` for incremental autoregressive decoding.
    The frequency axis can instead use explicit per-band tables built by
    `freq_tables` (see `TriangleRoPEBlock`).
    """

    def __init__(self, dim: int, base: float = 10_000.0):
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"RoPE requires a positive even dim, got {dim}.")
        self.dim = dim
        self.base = float(base)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_len = 0
        self.register_buffer("_cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("_sin_cached", torch.empty(0), persistent=False)

    def _get_cos_sin(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_len = offset + seq_len
        if (
            not torch.compiler.is_compiling()
            and self._cached_len >= total_len
            and self._cos_cached.device == device
            and self._cos_cached.dtype == dtype
        ):
            return (
                self._cos_cached[offset:total_len],
                self._sin_cached[offset:total_len],
            )
        positions = torch.arange(total_len, device=device, dtype=torch.float32)
        freqs = torch.outer(
            positions, self.inv_freq.to(device=device, dtype=torch.float32)
        )
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)
        if not torch.compiler.is_compiling():
            self._cos_cached = cos
            self._sin_cached = sin
            self._cached_len = total_len
        return cos[offset:total_len], sin[offset:total_len]

    def freq_tables(self, log_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build `[F, dim // 2]` cos/sin tables for the frequency axis.

        Rotary channel pairs are interleaved between two position systems so both
        cover the full range of rotation frequencies:
          - even pairs: log-frequency positions (semitones; pitch-invariant
            harmonic ratios),
          - odd pairs: band index (linear spacing; harmonic comb periodicity).
        """
        if log_positions.ndim != 1:
            raise ValueError(
                f"log_positions must be 1D [F], got shape {tuple(log_positions.shape)}."
            )
        bands = log_positions.shape[0]
        log_pos = log_positions.to(dtype=torch.float32, device=self.inv_freq.device)
        index_pos = torch.arange(bands, dtype=torch.float32, device=log_pos.device)
        pair_is_log = (
            torch.arange(self.dim // 2, device=log_pos.device) % 2 == 0
        )
        positions = torch.where(
            pair_is_log[None, :], log_pos[:, None], index_pos[:, None]
        )
        angles = positions * self.inv_freq.to(dtype=torch.float32)[None, :]
        return angles.cos(), angles.sin()

    def apply_freq_tables(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply precomputed `[F, dim // 2]` tables along dim 2 of `[B, T, F, dim]`."""
        if cos.shape[0] != a.shape[2]:
            raise ValueError(
                f"Frequency RoPE tables cover {cos.shape[0]} bands but input has "
                f"{a.shape[2]} bands."
            )
        cos = cos.to(dtype=a.dtype)[None, None, :, :]
        sin = sin.to(dtype=a.dtype)[None, None, :, :]
        return self._rotate_pair(a, b, cos, sin)

    @staticmethod
    def _rotate_pair(
        a: torch.Tensor,
        b: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a_r, a_i = a.chunk(2, dim=-1)
        b_r, b_i = b.chunk(2, dim=-1)
        return (
            a_r * cos - a_i * sin,
            a_i * cos + a_r * sin,
            b_r * cos - b_i * sin,
            b_i * cos + b_r * sin,
        )

    def apply_1d(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply 1D RoPE to `[B, H, T, head_dim]` or `[B, T, dim]` tensors."""
        if a.ndim == 4:
            cos, sin = self._get_cos_sin(a.shape[2], a.device, a.dtype, offset=offset)
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        elif a.ndim == 3:
            cos, sin = self._get_cos_sin(a.shape[1], a.device, a.dtype, offset=offset)
            cos = cos[None, :, :]
            sin = sin[None, :, :]
        else:
            raise ValueError(
                f"apply_1d expects 3D [B, T, D] or 4D [B, H, T, D] tensors, got {a.ndim}D."
            )
        return self._rotate_pair(a, b, cos, sin)

    def apply_time(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply RoPE along dim 1 (time) of `[B, T, F, tri_dim]` tensors."""
        cos, sin = self._get_cos_sin(a.shape[1], a.device, a.dtype, offset=offset)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        return self._rotate_pair(a, b, cos, sin)

    def apply_freq(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply RoPE along dim 2 (freq) of `[B, T, F, tri_dim]` tensors."""
        cos, sin = self._get_cos_sin(a.shape[2], a.device, a.dtype, offset=offset)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        return self._rotate_pair(a, b, cos, sin)

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        axis: str = "1d",
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if axis == "1d":
            return self.apply_1d(a, b, offset=offset)
        if axis == "time":
            return self.apply_time(a, b, offset=offset)
        if axis == "freq":
            return self.apply_freq(a, b, offset=offset)
        raise ValueError(f"axis must be '1d', 'time', or 'freq', got {axis!r}.")
