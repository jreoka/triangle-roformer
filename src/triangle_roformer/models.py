from __future__ import annotations

import math
import random
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .blocks_1d import TriangleRoFormerBlock1D, TriangleRoFormerState
from .blocks_2d import make_freq_triangle_block, make_time_triangle_block
from .layers import FineRateBoundaryBlock, GEGLUFusion
from .rope import log_frequency_positions


def _configure_dynamo_cache() -> None:
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
        torch._dynamo.config.cache_size_limit = max(
            int(getattr(torch._dynamo.config, "cache_size_limit", 8)), 64
        )
        if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
            torch._dynamo.config.accumulated_cache_size_limit = max(
                int(torch._dynamo.config.accumulated_cache_size_limit), 256
            )


class TriangleRoFormer(nn.Module):
    """1D Multi-Head Triangle-RoPE Transformer backbone for language, 1D audio,
    speech, and general sequence modeling.

    Expects `[B, T, dim]` and returns `[B, T, dim]`.

    Args:
        dim: Model hidden dimension.
        depth: Number of stacked `TriangleRoFormerBlock1D` layers.
        heads: Number of attention/contraction heads.
        causal: If True, uses exact autoregressive prefix contractions (with
            optional `states` caching for O(1)-memory decoding). If False,
            uses bidirectional O(T * H * head_dim^2) global contractions.
        head_dim: Per-head channel width (even integer; defaults to `dim // heads`).
        ffn_mult: SwiGLU FFN expansion ratio.
        rope_base: Rotary embedding base frequency.
        dropout: Dropout probability.
        drop_path: Linearly scaled stochastic depth rate across layers.
        use_checkpoint: Enable gradient checkpointing during training.
        chunk_size: Temporal chunk size for causal scan memory bounding.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int = 8,
        causal: bool = False,
        head_dim: int | None = None,
        ffn_mult: float = 3.0,
        rope_base: float = 10_000.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_checkpoint: bool = False,
        chunk_size: int = 64,
    ):
        super().__init__()
        if dim <= 0 or depth <= 0:
            raise ValueError(f"dim and depth must be positive, got dim={dim}, depth={depth}.")
        if not 0.0 <= drop_path < 1.0:
            raise ValueError(f"drop_path must be in [0, 1), got {drop_path}.")

        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.causal = causal
        self.drop_path = drop_path
        self.use_checkpoint = use_checkpoint

        res_scale = 1.0 / math.sqrt(2.0 * depth)
        self.layers = nn.ModuleList(
            TriangleRoFormerBlock1D(
                dim=dim,
                heads=heads,
                head_dim=head_dim,
                causal=causal,
                ffn_mult=ffn_mult,
                rope_base=rope_base,
                dropout=dropout,
                residual_scale=res_scale,
                chunk_size=chunk_size,
            )
            for _ in range(depth)
        )
        self.output_norm = nn.RMSNorm(dim)

    def compile_layers(self, mode: str = "default") -> None:
        _configure_dynamo_cache()
        for layer in self.layers:
            layer.compile(mode=mode, dynamic=False)

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
        mask_cpu = torch.rand(batch, 1, 1) < keep_prob
        if not torch.compiler.is_compiling():
            if not bool(mask_cpu.any()):
                return 0.0
            if bool(mask_cpu.all()):
                return 1.0 / keep_prob
        mask = mask_cpu.to(device=device, dtype=dtype, non_blocking=True)
        return mask / keep_prob

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        states: Sequence[TriangleRoFormerState | None] | None = None,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[TriangleRoFormerState | None]]:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input [B, T, D], got {tuple(x.shape)}.")
        if states is not None and len(states) != self.depth:
            raise ValueError(
                f"Expected {self.depth} layer states, got {len(states)}."
            )
        if torch.is_autocast_enabled(x.device.type):
            x = x.to(torch.get_autocast_dtype(x.device.type))

        batch = x.shape[0]
        should_checkpoint = (
            self.use_checkpoint
            and torch.is_grad_enabled()
            and x.requires_grad
            and not return_states
            and states is None
        )
        denom = max(1, self.depth - 1)
        next_states: list[TriangleRoFormerState | None] = []

        h = x
        for idx, layer in enumerate(self.layers):
            layer_state = states[idx] if states is not None else None
            p_drop = self.drop_path * (idx / denom) if self.depth > 1 else 0.0
            keep = (
                None
                if (return_states or layer_state is not None)
                else self._drop_keep_mask(batch, p_drop, h.device, h.dtype)
            )
            if isinstance(keep, float) and keep == 0.0:
                if return_states:
                    next_states.append(layer_state)
                continue

            if return_states:
                out, new_state = layer(
                    h,
                    attention_mask=attention_mask,
                    state=layer_state,
                    return_state=True,
                )
                next_states.append(new_state)
            elif should_checkpoint:
                if attention_mask is None:
                    out = checkpoint(layer, h, use_reentrant=False)
                else:
                    out = checkpoint(
                        layer, h, attention_mask, use_reentrant=False
                    )
            else:
                out = layer(
                    h,
                    attention_mask=attention_mask,
                    state=layer_state,
                    return_state=False,
                )

            h = out if keep is None else h + (out - h) * keep

        h = self.output_norm(h)
        if return_states:
            return h, next_states
        return h


class TriangleRoFormerLM(nn.Module):
    """Drop-in Language / Discrete-Token Model built on `TriangleRoFormer`.

    Expects integer `input_ids` of shape `[B, T]` and returns vocabulary logits
    of shape `[B, T, vocab_size]`. Defaults to `causal=True` for autoregressive
    language or discrete audio-codec modeling.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        heads: int = 8,
        causal: bool = True,
        head_dim: int | None = None,
        ffn_mult: float = 3.0,
        rope_base: float = 10_000.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_checkpoint: bool = False,
        chunk_size: int = 64,
        tie_weights: bool = True,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")
        self.vocab_size = vocab_size
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.emb_drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.backbone = TriangleRoFormer(
            dim=dim,
            depth=depth,
            heads=heads,
            causal=causal,
            head_dim=head_dim,
            ffn_mult=ffn_mult,
            rope_base=rope_base,
            dropout=dropout,
            drop_path=drop_path,
            use_checkpoint=use_checkpoint,
            chunk_size=chunk_size,
        )
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        nn.init.normal_(self.token_emb.weight, std=1.0 / math.sqrt(dim))
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight
        else:
            nn.init.normal_(self.lm_head.weight, std=1.0 / math.sqrt(dim))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        states: Sequence[TriangleRoFormerState | None] | None = None,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[TriangleRoFormerState | None]]:
        x = self.emb_drop(self.token_emb(input_ids))
        if return_states:
            h, next_states = self.backbone(
                x,
                attention_mask=attention_mask,
                states=states,
                return_states=True,
            )
            return self.lm_head(h), next_states
        h = self.backbone(
            x,
            attention_mask=attention_mask,
            states=states,
            return_states=False,
        )
        return self.lm_head(h)


class TriangleRoFormer2D(nn.Module):
    """2D Axial Time-Frequency Triangle-RoPE Backbone for audio spectrograms
    and 2D feature grids `[B, T, F, D]`.

    When `use_boundary_blocks=True` (default):
      1. Fine-rate PreBlock (`ConvNeXtV2TemporalMixer` + time & freq Triangle-RoPE)
         producing fine-rate skip representation `X'`.
      2. `depth` stacked axial Triangle-RoPE pairs alternating time-axis `T_l` and
         frequency-axis `B_l` (`H^(l) = B_l(T_l(H^(l-1)))`).
      3. `GEGLUFusion` combining deep context `Y'` and fine-rate skip `X'`.
      4. Fine-rate PostBlock + terminal `RMSNorm`.

    Multi-scale layer schedule (non-causal), for layer index `l`:
      - Time `T_l`: odd `l` use a local windowed contraction over `time_window`
        frames (half-shift on every other windowed layer); even `l` contract over
        the whole chunk. Each layer pairs one local and one global axis.
      - Freq `B_l`: even `l` pool their band-to-band affinity over local windows,
        alternating `short_freq_window` and `freq_window` frames (half-shift on
        every other pair); odd `l` pool over the whole chunk.

    `band_freqs_hz` gives each band's center frequency. Frequency-axis RoPE uses
    log-frequency (semitone) positions on half of its rotary pairs so harmonic
    ratios are fixed offsets regardless of pitch, and band index on the other half.

    Separation-oriented options (all blocks):
      - `soft_norm` / `norm_floor_init`: confidence-preserving normalization of
        time transfer matrices and frequency affinity logits (learnable floor).
      - `attn_sink`: null logit in every frequency softmax.
      - `band_embedding`: learned zero-initialized `[F, dim]` absolute band
        embedding added to the input (off by default; unnecessary when the
        front end already uses per-band projections).
      - `forward(x, frame_mask=...)`: `[B, T]` mask (1 = real frame) that removes
        padded frames from every time-pooled statistic (chunk-wide, windowed,
        causal prefix, and the boundary blocks' GRN).

    When `causal=True`:
      - All time-axis blocks run causally over the prefix (frame `t` only depends
        on frames `<= t`); with `causal_decay=True` the prefix is an exponentially
        decayed mean with learnable per-channel decay.
      - Frequency-axis blocks pool their affinity over the same kind of decayed
        causal prefix (`CausalFreqTriangleRoPEBlock`) instead of the whole chunk.
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

        freq_schedule = [
            self._freq_window_for_layer(idx, causal, freq_window, short_freq_window)
            for idx in range(depth)
        ]
        time_schedule = [
            self._time_window_for_layer(idx, causal, time_window)
            for idx in range(depth)
        ]
        self.freq_layers = nn.ModuleList(
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
            for f_win, f_shift in freq_schedule
        )
        self.time_layers = nn.ModuleList(
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
            for t_win, t_shift in time_schedule
        )

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
        shift = time_window // 2 if (idx // 2) % 2 == 1 else 0
        return time_window, shift

    def compile_layers(self, mode: str = "default") -> None:
        _configure_dynamo_cache()
        if self.pre_block is not None:
            self.pre_block.compile_layers(mode=mode)
        for unit in (*self.freq_layers, *self.time_layers):
            unit.compile(mode=mode, dynamic=False)
        if self.geglu_fusion is not None:
            self.geglu_fusion.compile(mode=mode, dynamic=False)
        if self.post_block is not None:
            self.post_block.compile_layers(mode=mode)

    @staticmethod
    def _run_module(
        module: nn.Module,
        x: torch.Tensor,
        use_checkpoint: bool,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not use_checkpoint:
            return module(x, frame_mask)
        return checkpoint(module, x, frame_mask, use_reentrant=False)

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
        statistics. Outputs at padded frames are unspecified."""
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
        should_checkpoint = (
            self.use_checkpoint and torch.is_grad_enabled() and x.requires_grad
        )

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

            if should_checkpoint:
                y_fused = checkpoint(
                    self.geglu_fusion, x_fine_skip, h, use_reentrant=False
                )
            else:
                y_fused = self.geglu_fusion(x_fine_skip, h)

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
