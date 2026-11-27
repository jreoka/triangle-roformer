"""triangle-roformer: Linear-complexity Triangle RoFormer architecture for 1D & 2D modeling."""

from .blocks_1d import TriangleRoFormerBlock1D, TriangleRoFormerState
from .blocks_2d import (
    GlobalFreqTriangleRoPEBlock,
    TimeTriangleRoPEBlock,
    TriangleRoPEBlock,
    WindowedFreqTriangleRoPEBlock,
    make_freq_triangle_block,
)
from .layers import ConvNeXtV2TemporalMixer, FineRateBoundaryBlock, GEGLUFusion
from .models import TriangleRoFormer, TriangleRoFormer2D, TriangleRoFormerLM
from .rope import RotaryEmbedding

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "RotaryEmbedding",
    "TriangleRoFormerState",
    "TriangleRoFormerBlock1D",
    "TriangleRoFormer",
    "TriangleRoFormerLM",
    "TriangleRoPEBlock",
    "TimeTriangleRoPEBlock",
    "GlobalFreqTriangleRoPEBlock",
    "WindowedFreqTriangleRoPEBlock",
    "make_freq_triangle_block",
    "ConvNeXtV2TemporalMixer",
    "FineRateBoundaryBlock",
    "GEGLUFusion",
    "TriangleRoFormer2D",
]
