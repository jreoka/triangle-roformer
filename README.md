<img width="2752" height="1536" alt="Gemini_Generated_Image_w22x39w22x39w22x" src="https://github.com/user-attachments/assets/b1538824-8722-4a09-86c5-98a951093c34" />

# triangle-roformer

An installable PyTorch library implementing **TriangleRoFormer2D** — a 2D axial time-frequency backbone for audio source separation and spectrogram modeling combining:

- **Multi-scale linear-time Triangle-RoPE** multiplicative blocks + **SwiGLU** FFNs,
- **Regular multi-head RoPE softmax attention** (`TimeRoPEAttentionBlock` & per-frame `FreqRoPEAttentionBlock`) on every 4th layer (`l % 4 == 3`) for sharp within-clip recall and instantaneous harmonic routing, and
- **Modern Continuous Hopfield Network (`HopfieldFFN`)** FFNs inside the frequency RoPE attention blocks for learned timbral/harmonic prototype retrieval alongside SwiGLU everywhere else.

## Installation

```bash
pip install ./triangle-roformer
# or editable install for development:
pip install -e ./triangle-roformer
```

Requires Python `>=3.10` and `torch>=2.4.0` (uses `nn.RMSNorm` and `F.scaled_dot_product_attention`).

## Quick Start

```python
import torch
from triangle_roformer import TriangleRoFormer2D

# Center frequency (Hz) of each of the F band tokens, low to high.
band_freqs_hz = [...]  # len == 124 here

encoder = TriangleRoFormer2D(
    band_freqs_hz,
    dim=256,
    depth=14,
    tri_dim=128,          # or heads=64 (in Triangle-RoPE blocks, heads = tri_dim // 2)
    causal=False,         # False for offline separation, True for causal time axis
    freq_window=32,       # long local pooling window for freq affinities (frames)
    short_freq_window=8,  # short local pooling window for freq affinities (frames)
    time_window=64,       # local window for windowed time contractions (frames)
    ffn_mult=3.0,
    dropout=0.0,
    drop_path=0.08,
    use_checkpoint=True,
    use_boundary_blocks=True,
    hopfield_patterns=256,
    hopfield_steps=2,
)

tokens = torch.randn(1, 64, 124, 256)  # [B, T, F, D]
out = encoder(tokens)                  # [1, 64, 124, 256]
```

### Fixed Hybrid Layer Schedule

| layer `l` | time block `T_l` | freq block `B_l` |
|---|---|---|
| `l % 4 == 0` | chunk-wide Triangle-RoPE + SwiGLU | windowed Triangle-RoPE (`short_freq_window` / `freq_window`) + SwiGLU |
| `l % 4 == 1` | windowed Triangle-RoPE (`time_window`) + SwiGLU | chunk-wide Triangle-RoPE + SwiGLU |
| `l % 4 == 2` | chunk-wide Triangle-RoPE + SwiGLU | windowed Triangle-RoPE (`freq_window` / `short_freq_window`) + SwiGLU |
| `l % 4 == 3` | **TimeRoPEAttentionBlock** + SwiGLU | **FreqRoPEAttentionBlock** + **HopfieldFFN** |

### Log-frequency RoPE

Both Triangle-RoPE and regular RoPE attention on the frequency axis rotate half of their RoPE pairs by **log-frequency positions** (`12 * log2(hz / 20)`, semitones), so a harmonic ratio such as an octave is the same relative offset for every fundamental, and the other half by band index (linear harmonic-comb spacing).

### Separation-oriented options

| option | what it does |
|---|---|
| `soft_norm=True`, `norm_floor_init=0.5` | Time transfer matrices and frequency affinity logits are divided by `sqrt(rms² + floor²/n)` rather than by their own RMS (`n` = frames pooled, `floor` learnable per channel). |
| `attn_sink=True` | Adds a learnable per-head null/sink option with zero value to frequency Triangle-RoPE, regular RoPE attention (`sink_k`), and `HopfieldFFN` (`sink_logit`). |
| `causal_decay=True` | Causal mode only: exponentially decayed prefix with learnable log-spaced per-channel half-lives. |
| `band_embedding=False` | Optional zero-initialized `[F, dim]` absolute band embedding. |

### Padding mask

```python
# frame_mask: [B, T], 1 = real frame. Padded frames are excluded from every
# time-pooled statistic, time attention, and boundary-block GRN.
out = encoder(tokens, frame_mask=frame_mask)
```

## Running Tests

```bash
pip install -e "./triangle-roformer[dev]"
pytest ./triangle-roformer/tests
```
