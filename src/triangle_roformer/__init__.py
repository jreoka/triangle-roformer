"""triangle-roformer: Linear-complexity Triangle RoFormer architecture for 1D & 2D modeling."""

from .blocks_1d import TriangleRoFormerBlock1D, TriangleRoFormerState
from .blocks_2d import (
    CausalFreqTriangleRoPEBlock,
    GlobalFreqTriangleRoPEBlock,
    TimeTriangleRoPEBlock,
    TriangleRoPEBlock,
    WindowedFreqTriangleRoPEBlock,
    WindowedTimeTriangleRoPEBlock,
    make_freq_triangle_block,
    make_time_triangle_block,
)
from .layers import ConvNeXtV2TemporalMixer, FineRateBoundaryBlock, GEGLUFusion
from .models import TriangleRoFormer, TriangleRoFormer2D, TriangleRoFormerLM
from .rope import RotaryEmbedding, log_frequency_positions

__version__ = "0.3.1"

__all__ = [
    "__version__",
    "RotaryEmbedding",
    "log_frequency_positions",
    "TriangleRoFormerState",
    "TriangleRoFormerBlock1D",
    "TriangleRoFormer",
    "TriangleRoFormerLM",
    "TriangleRoPEBlock",
    "TimeTriangleRoPEBlock",
    "WindowedTimeTriangleRoPEBlock",
    "GlobalFreqTriangleRoPEBlock",
    "WindowedFreqTriangleRoPEBlock",
    "CausalFreqTriangleRoPEBlock",
    "make_freq_triangle_block",
    "make_time_triangle_block",
    "ConvNeXtV2TemporalMixer",
    "FineRateBoundaryBlock",
    "GEGLUFusion",
    "TriangleRoFormer2D",
]
