# triangle-roformer

An installable PyTorch library implementing the **Triangle-RoPE Transformer** — a linear-complexity relative-RoPE triangle multiplicative update architecture for **1D sequences** (language models, 1D audio/speech, time series) and **2D time-frequency grids** (audio source separation, spectrograms).

## Installation

Unzip the archive and install with `pip`:

```bash
pip install ./triangle-roformer
# or editable install for development:
pip install -e ./triangle-roformer
```

Requires Python `>=3.10` and `torch>=2.4.0` (uses `nn.RMSNorm`).

## Quick Start

### 1. 1D Sequence Backbone (Audio, Speech, Language, Time Series)

```python
import torch
from triangle_roformer import TriangleRoFormer

model = TriangleRoFormer(
    dim=512,
    depth=12,
    heads=8,
    causal=True,        # True for autoregressive, False for bidirectional
    ffn_mult=3.0,
    dropout=0.0,
    drop_path=0.08,
    use_checkpoint=False,
)

x = torch.randn(2, 128, 512)
y = model(x)  # [2, 128, 512]
```

#### Incremental $O(1)$-Memory Autoregressive Decoding

```python
# Prefill prompt
y_prompt, states = model(x[:, :64], return_states=True)

# Decode 1 token at a time with constant state size
y_next, states = model(x[:, 64:65], states=states, return_states=True)
```

### 2. Language / Discrete Token Model

```python
import torch
from triangle_roformer import TriangleRoFormerLM

lm = TriangleRoFormerLM(
    vocab_size=32_000,
    dim=512,
    depth=12,
    heads=8,
    causal=True,
)

input_ids = torch.randint(0, 32_000, (2, 128))
logits = lm(input_ids)  # [2, 128, 32000]
```

### 3. 2D Axial Time-Frequency Architecture (Spectrograms / MVSEP)

```python
import torch
from triangle_roformer import TriangleRoFormer2D

# Center frequency (Hz) of each of the F band tokens, low to high.
band_freqs_hz = [...]  # len == 124 here

encoder = TriangleRoFormer2D(
    band_freqs_hz,
    dim=256,
    depth=14,
    tri_dim=128,          # or heads=64 (in 2D blocks, heads = tri_dim // 2)
    causal=False,         # False for offline separation, True for causal time axis
    freq_window=32,       # long local pooling window for freq affinities (frames)
    short_freq_window=8,  # short local pooling window for freq affinities (frames)
    time_window=64,       # local window for windowed time contractions (frames)
    ffn_mult=3.0,
    dropout=0.0,
    drop_path=0.08,
    use_checkpoint=True,
    use_boundary_blocks=True,
)

tokens = torch.randn(1, 64, 124, 256)  # [B, T, F, D]
out = encoder(tokens)                  # [1, 64, 124, 256]
```

#### Multi-scale layer schedule

Every axial layer pairs one local and one chunk-wide contraction:

| layer `l` | time block `T_l` | freq block `B_l` |
|---|---|---|
| even | chunk-wide time contraction | windowed affinity, alternating `short_freq_window` / `freq_window` |
| odd  | windowed time contraction over `time_window` frames | chunk-wide affinity |

Windowed layers alternate a half-window shift so window seams move between layers.
In causal mode, time blocks use the exact full-prefix contraction and freq blocks use
frame-local affinity.

#### Log-frequency RoPE

Frequency-axis blocks rotate half of their RoPE pairs by **log-frequency positions**
(`12 * log2(hz / 20)`, semitones), so a harmonic ratio such as an octave is the same
relative offset for every fundamental, and the other half by band index (linear
harmonic-comb spacing). This is why `TriangleRoFormer2D` requires `band_freqs_hz`.

## Running Tests

```bash
pip install -e "./triangle-roformer[dev]"
pytest ./triangle-roformer/tests
```
