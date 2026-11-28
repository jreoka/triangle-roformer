"""triangle-roformer: 2D Axial Time-Frequency Triangle-RoPE Transformer for audio."""

from .blocks_2d import (
    CausalFreqTriangleRoPEBlock,
    FreqRoPEAttentionBlock,
    GlobalFreqTriangleRoPEBlock,
    RoPEAttentionBlock,
    TimeRoPEAttentionBlock,
    TimeTriangleRoPEBlock,
    TriangleRoPEBlock,
    WindowedFreqTriangleRoPEBlock,
    WindowedTimeTriangleRoPEBlock,
    make_freq_triangle_block,
    make_time_triangle_block,
)
from .hopfield import HopfieldFFN
from .layers import ConvNeXtV2TemporalMixer, FineRateBoundaryBlock, GEGLUFusion
from .models import TriangleRoFormer2D
from .rope import RotaryEmbedding, log_frequency_positions

__version__ = "0.4.0"

__all__ = [
    "__version__",
    "RotaryEmbedding",
    "log_frequency_positions",
    "HopfieldFFN",
    "TriangleRoPEBlock",
    "TimeTriangleRoPEBlock",
    "WindowedTimeTriangleRoPEBlock",
    "GlobalFreqTriangleRoPEBlock",
    "WindowedFreqTriangleRoPEBlock",
    "CausalFreqTriangleRoPEBlock",
    "RoPEAttentionBlock",
    "TimeRoPEAttentionBlock",
    "FreqRoPEAttentionBlock",
    "make_freq_triangle_block",
    "make_time_triangle_block",
    "ConvNeXtV2TemporalMixer",
    "FineRateBoundaryBlock",
    "GEGLUFusion",
    "TriangleRoFormer2D",
]
