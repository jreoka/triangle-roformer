from __future__ import annotations

import math
import random
from typing import Sequence

import torch
import torch.nn as nn

from ._compile import compile_unit, configure_dynamo_cache, run_unit
from .blocks_2d import (
    FreqRoPEAttentionBlock,
    TimeRoPEAttentionBlock,
    make_freq_triangle_block,
    make_time_triangle_block,
)
from .layers import FineRateBoundaryBlock, GEGLUFusion
from .rope import log_frequency_positions


def _configure_dynamo_cache() -> None:
    configure_dynamo_cache()


class TriangleRoFormer2D(nn.Module):
    """2D Axial Time-Frequency Triangle-RoPE Backbone for audio spectrograms
    and 2D feature grids `[B, T, F, D]`.

    Fixed hybrid architecture combining:
      - Multi-scale linear-time **Triangle-RoPE** multiplicative blocks with
        SwiGLU FFNs on layers `l % 4 != 3`, and
      - Regular multi-head **RoPE softmax attention** on every 4th layer
        (`l % 4 == 3`): `TimeRoPEAttentionBlock` (SwiGLU FFN) for sharp
        frame-to-frame recall and per-frame `FreqRoPEAttentionBlock`
        (`HopfieldFFN`) for instantaneous harmonic routing + timbral prototype
        retrieval.

    When `use_boundary_blocks=True` (default):
      1. Fine-rate PreBlock (`ConvNeXtV2TemporalMixer` + time & freq Triangle-RoPE)
         producing fine-rate skip representation `X'`.
      2. `depth` stacked axial pairs alternating time-axis `T_l` and
         frequency-axis `B_l` (`H^(l) = B_l(T_l(H^(l-1)))`).
      3. `GEGLUFusion` combining deep context `Y'` and fine-rate skip `X'`.
      4. Fine-rate PostBlock + terminal `RMSNorm`.

    `band_freqs_hz` gives each band's center frequency. Both Triangle-RoPE and
    regular RoPE attention on the frequency axis use interleaved log-frequency
    (semitone) positions on half of their rotary pairs so harmonic ratios are
    fixed offsets regardless of pitch, and band index on the other half.
    """

    def __init__(
        self,
        band_freqs_hz: Sequence[float],
        dim: int = 256,
        depth: int = 14,
        tri_dim: int | None = None,
        heads: int | None = None,
        causal: bool = False,
        freq_window: int = 32,
        short_freq_window: int = 8,
        time_window: int = 64,
        ffn_mult: float = 3.0,
        dropout: float = 0.0,
        drop_path: float = 0.08,
        use_checkpoint: bool = False,
        use_boundary_blocks: bool = True,
        time_rope_base: float = 10_000.0,
        freq_rope_base: float = 1_000.0,
        chunk_size: int = 32,
        soft_norm: bool = True,
        norm_floor_init: float = 0.5,
        attn_sink: bool = True,
        causal_decay: bool = True,
        band_embedding: bool = False,
        attn_heads: int | None = None,
        hopfield_heads: int | None = None,
        hopfield_patterns: int = 256,
        hopfield_steps: int = 2,
        hopfield_step_size: float = 1.0,
    ):
        super().__init__()
        if dim <= 0 or depth <= 0:
            raise ValueError(f"dim and depth must be positive, got dim={dim}, depth={depth}.")
        for name, value in (
            ("freq_window", freq_window),
            ("short_freq_window", short_freq_window),
            ("time_window", time_window),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if not 0.0 <= drop_path < 1.0:
            raise ValueError(f"drop_path must be in [0, 1), got {drop_path}.")

        freq_positions = log_frequency_positions(band_freqs_hz)
        self.num_bands = int(freq_positions.shape[0])
        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.drop_path = drop_path
        self.use_checkpoint = use_checkpoint
        self.use_boundary_blocks = use_boundary_blocks

        total_axial_pairs = depth + (2 if use_boundary_blocks else 0)
        res_scale = 1.0 / math.sqrt(2.0 * total_axial_pairs)
        half_shift = 0 if causal else (freq_window // 2)
        block_opts = dict(
            soft_norm=soft_norm,
            norm_floor_init=norm_floor_init,
            causal_decay=causal_decay,
        )
        hopfield_opts = dict(
            hopfield_heads=hopfield_heads,
            hopfield_patterns=hopfield_patterns,
            hopfield_steps=hopfield_steps,
            hopfield_step_size=hopfield_step_size,
        )

        if band_embedding:
            self.band_emb: nn.Parameter | None = nn.Parameter(
                torch.zeros(self.num_bands, dim)
            )
        else:
            self.register_parameter("band_emb", None)

        if use_boundary_blocks:
            self.pre_block: FineRateBoundaryBlock | None = FineRateBoundaryBlock(
                dim=dim,
                tri_dim=tri_dim,
                heads=heads,
                causal=causal,
                ffn_mult=ffn_mult,
                freq_window=freq_window,
                freq_window_shift=0,
                dropout=dropout,
                residual_scale=res_scale,
                freq_positions=freq_positions,
                chunk_size=chunk_size,
                attn_sink=attn_sink,
                **block_opts,
            )
        else:
            self.pre_block = None

        freq_layers: list[nn.Module] = []
        time_layers: list[nn.Module] = []
        for idx in range(depth):
            if idx % 4 == 3:
                time_layers.append(
                    TimeRoPEAttentionBlock(
                        dim=dim,
                        heads=attn_heads,
                        causal=causal,
                        ffn_mult=ffn_mult,
                        rope_base=time_rope_base,
                        dropout=dropout,
                        residual_scale=res_scale,
                        attn_sink=attn_sink,
                    )
                )
                freq_layers.append(
                    FreqRoPEAttentionBlock(
                        dim=dim,
                        heads=attn_heads,
                        rope_base=freq_rope_base,
                        dropout=dropout,
                        residual_scale=res_scale,
                        freq_positions=freq_positions,
                        attn_sink=attn_sink,
                        **hopfield_opts,
                    )
                )
            else:
                t_win, t_shift = self._time_window_for_layer(idx, causal, time_window)
                time_layers.append(
                    make_time_triangle_block(
                        dim=dim,
                        tri_dim=tri_dim,
                        heads=heads,
                        causal=causal,
                        chunk_size=chunk_size,
                        ffn_mult=ffn_mult,
                        rope_base=time_rope_base,
                        time_window=t_win,
                        window_shift=t_shift,
                        dropout=dropout,
                        residual_scale=res_scale,
                        **block_opts,
                    )
                )
                f_win, f_shift = self._freq_window_for_layer(
                    idx, causal, freq_window, short_freq_window
                )
                freq_layers.append(
                    make_freq_triangle_block(
                        dim=dim,
                        tri_dim=tri_dim,
                        heads=heads,
                        ffn_mult=ffn_mult,
                        rope_base=freq_rope_base,
                        freq_window=f_win,
                        window_shift=f_shift,
                        dropout=dropout,
                        residual_scale=res_scale,
                        freq_positions=freq_positions,
                        causal=causal,
                        chunk_size=chunk_size,
                        attn_sink=attn_sink,
                        **block_opts,
                    )
                )

        self.time_layers = nn.ModuleList(time_layers)
        self.freq_layers = nn.ModuleList(freq_layers)

        if use_boundary_blocks:
            self.geglu_fusion: GEGLUFusion | None = GEGLUFusion(
                dim=dim,
                ffn_mult=min(2.0, ffn_mult),
                dropout=dropout,
                residual_scale=res_scale,
            )
            self.post_block: FineRateBoundaryBlock | None = FineRateBoundaryBlock(
                dim=dim,
                tri_dim=tri_dim,
                heads=heads,
                causal=causal,
                ffn_mult=ffn_mult,
                freq_window=freq_window,
                freq_window_shift=half_shift,
                dropout=dropout,
                residual_scale=res_scale,
                freq_positions=freq_positions,
                chunk_size=chunk_size,
                attn_sink=attn_sink,
                **block_opts,
            )
        else:
            self.geglu_fusion = None
            self.post_block = None

        self.output_norm = nn.RMSNorm(dim)

    @staticmethod
    def _freq_window_for_layer(
        idx: int, causal: bool, long_window: int, short_window: int
    ) -> tuple[int, int]:
        """(window, shift) for frequency layer `idx`; window 0 = chunk-wide.
        Causal layers ignore it (decayed causal prefix)."""
        if causal:
            return 0, 0
        if idx % 2 == 1:
            return 0, 0
        window = short_window if (idx // 2) % 2 == 0 else long_window
        shift = window // 2 if (idx // 4) % 2 == 1 else 0
        return window, shift

    @staticmethod
    def _time_window_for_layer(
        idx: int, causal: bool, time_window: int
    ) -> tuple[int, int]:
        """(window, shift) for time layer `idx`; window 0 = chunk-wide."""
        if causal or idx % 2 == 0:
            return 0, 0
        # Windowed time layers occur at idx % 4 == 1 (idx % 4 == 3 is RoPE attention),
        # so alternate shift by idx // 4 to actually move window seams across layers.
        shift = time_window // 2 if (idx // 4) % 2 == 1 else 0
        return time_window, shift

    def compile_layers(
        self, mode: str = "default", dynamic: bool | None = False
    ) -> None:
        _configure_dynamo_cache()
        if self.pre_block is not None:
            self.pre_block.compile_layers(mode=mode, dynamic=dynamic)
        for unit in (*self.freq_layers, *self.time_layers):
            compile_unit(unit, mode=mode, dynamic=dynamic)
        if self.geglu_fusion is not None:
            compile_unit(self.geglu_fusion, mode=mode, dynamic=dynamic)
        if self.post_block is not None:
            self.post_block.compile_layers(mode=mode, dynamic=dynamic)

    @staticmethod
    def _run_module(
        module: nn.Module,
        x: torch.Tensor,
        use_checkpoint: bool,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return run_unit(module, use_checkpoint, x, frame_mask)

    def _drop_keep_mask(
        self,
        batch: int,
        drop_prob: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | float | None:
        if not self.training or drop_prob <= 0.0:
            return None
        keep_prob = 1.0 - drop_prob
        if batch == 1 and not torch.compiler.is_compiling():
            return 0.0 if random.random() < drop_prob else (1.0 / keep_prob)
        mask_cpu = torch.rand(batch, 1, 1, 1) < keep_prob
        if not torch.compiler.is_compiling():
            if not bool(mask_cpu.any()):
                return 0.0
            if bool(mask_cpu.all()):
                return 1.0 / keep_prob
        mask = mask_cpu.to(device=device, dtype=dtype, non_blocking=True)
        return mask / keep_prob

    def forward(
        self, x: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """`x`: `[B, T, F, D]`. `frame_mask`: optional `[B, T]` (bool or float,
        1 = real frame); padded frames are excluded from all time-pooled
        statistics and time attention. Outputs at padded frames are unspecified."""
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input [B, T, F, D], got {tuple(x.shape)}.")
        if x.shape[2] != self.num_bands:
            raise ValueError(
                f"Model was built for {self.num_bands} bands (len(band_freqs_hz)) "
                f"but input has {x.shape[2]}."
            )
        if frame_mask is not None:
            if frame_mask.shape != x.shape[:2]:
                raise ValueError(
                    f"frame_mask must be [B, T] = {tuple(x.shape[:2])}, "
                    f"got {tuple(frame_mask.shape)}."
                )
            frame_mask = frame_mask.to(device=x.device, dtype=torch.float32)
        batch = x.shape[0]
        if torch.is_autocast_enabled(x.device.type):
            x = x.to(torch.get_autocast_dtype(x.device.type))
        if self.band_emb is not None:
            x = x + self.band_emb.to(dtype=x.dtype)
        # use_reentrant=False works even when input x.requires_grad is False
        # (e.g. raw features / frozen frontend), so do not silently disable checkpointing.
        should_checkpoint = self.use_checkpoint and torch.is_grad_enabled()

        if self.use_boundary_blocks:
            assert self.pre_block is not None
            assert self.geglu_fusion is not None
            assert self.post_block is not None
            total_blocks = max(1, 2 * (self.depth + 2) - 1)

            keep_pre_t = self._drop_keep_mask(batch, 0.0, x.device, x.dtype)
            keep_pre_f = self._drop_keep_mask(
                batch, self.drop_path * 1.0 / total_blocks, x.device, x.dtype
            )
            x_fine_skip = self.pre_block(
                x,
                should_checkpoint=should_checkpoint,
                keep_t=keep_pre_t,
                keep_f=keep_pre_f,
                frame_mask=frame_mask,
            )

            h = x_fine_skip
            for layer_idx, (time_layer, freq_layer) in enumerate(
                zip(self.time_layers, self.freq_layers)
            ):
                block_offset = 2 * (layer_idx + 1)
                p_time = self.drop_path * block_offset / total_blocks
                keep_t = self._drop_keep_mask(batch, p_time, h.device, h.dtype)
                if not (isinstance(keep_t, float) and keep_t == 0.0):
                    time_out = self._run_module(time_layer, h, should_checkpoint, frame_mask)
                    h = time_out if keep_t is None else h + (time_out - h) * keep_t

                p_freq = self.drop_path * (block_offset + 1) / total_blocks
                keep_f = self._drop_keep_mask(batch, p_freq, h.device, h.dtype)
                if not (isinstance(keep_f, float) and keep_f == 0.0):
                    freq_out = self._run_module(freq_layer, h, should_checkpoint, frame_mask)
                    h = freq_out if keep_f is None else h + (freq_out - h) * keep_f

            y_fused = run_unit(
                self.geglu_fusion, should_checkpoint, x_fine_skip, h
            )

            post_offset = 2 * (self.depth + 1)
            keep_post_t = self._drop_keep_mask(
                batch,
                self.drop_path * post_offset / total_blocks,
                y_fused.device,
                y_fused.dtype,
            )
            keep_post_f = self._drop_keep_mask(
                batch,
                self.drop_path * (post_offset + 1) / total_blocks,
                y_fused.device,
                y_fused.dtype,
            )
            u = self.post_block(
                y_fused,
                should_checkpoint=should_checkpoint,
                keep_t=keep_post_t,
                keep_f=keep_post_f,
                frame_mask=frame_mask,
            )
            return self.output_norm(u)

        total_blocks = max(1, 2 * self.depth - 1)
        h = x
        for layer_idx, (time_layer, freq_layer) in enumerate(
            zip(self.time_layers, self.freq_layers)
        ):
            block_offset = 2 * layer_idx
            p_time = self.drop_path * block_offset / total_blocks
            keep_t = self._drop_keep_mask(batch, p_time, h.device, h.dtype)
            if not (isinstance(keep_t, float) and keep_t == 0.0):
                time_out = self._run_module(time_layer, h, should_checkpoint, frame_mask)
                h = time_out if keep_t is None else h + (time_out - h) * keep_t

            p_freq = self.drop_path * (block_offset + 1) / total_blocks
            keep_f = self._drop_keep_mask(batch, p_freq, h.device, h.dtype)
            if not (isinstance(keep_f, float) and keep_f == 0.0):
                freq_out = self._run_module(freq_layer, h, should_checkpoint, frame_mask)
                h = freq_out if keep_f is None else h + (freq_out - h) * keep_f

        return self.output_norm(h)
