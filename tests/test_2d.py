from __future__ import annotations

import pytest
import torch

from triangle_roformer import (
    GlobalFreqTriangleRoPEBlock,
    TimeTriangleRoPEBlock,
    TriangleRoFormer2D,
    WindowedFreqTriangleRoPEBlock,
    WindowedTimeTriangleRoPEBlock,
    log_frequency_positions,
)


def _band_freqs(bands: int) -> list[float]:
    return [50.0 * (1.35**i) for i in range(bands)]


def test_2d_non_causal_forward_backward() -> None:
    model = TriangleRoFormer2D(
        _band_freqs(12),
        dim=64,
        depth=4,
        tri_dim=32,
        causal=False,
        freq_window=8,
        short_freq_window=4,
        time_window=6,
        use_boundary_blocks=True,
    )
    x = torch.randn(1, 16, 12, 64, requires_grad=True)
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_2d_layer_schedule() -> None:
    model = TriangleRoFormer2D(
        _band_freqs(8),
        dim=32,
        depth=8,
        tri_dim=16,
        freq_window=32,
        short_freq_window=8,
        time_window=64,
    )
    time_kinds = [
        (type(layer), getattr(layer, "time_window", 0), getattr(layer, "time_window_shift", 0))
        for layer in model.time_layers
    ]
    assert time_kinds == [
        (TimeTriangleRoPEBlock, 0, 0),
        (WindowedTimeTriangleRoPEBlock, 64, 0),
        (TimeTriangleRoPEBlock, 0, 0),
        (WindowedTimeTriangleRoPEBlock, 64, 32),
        (TimeTriangleRoPEBlock, 0, 0),
        (WindowedTimeTriangleRoPEBlock, 64, 0),
        (TimeTriangleRoPEBlock, 0, 0),
        (WindowedTimeTriangleRoPEBlock, 64, 32),
    ]
    freq_kinds = [
        (type(layer), layer.freq_window, layer.window_shift) for layer in model.freq_layers
    ]
    assert freq_kinds == [
        (WindowedFreqTriangleRoPEBlock, 8, 0),
        (GlobalFreqTriangleRoPEBlock, 0, 0),
        (WindowedFreqTriangleRoPEBlock, 32, 0),
        (GlobalFreqTriangleRoPEBlock, 0, 0),
        (WindowedFreqTriangleRoPEBlock, 8, 4),
        (GlobalFreqTriangleRoPEBlock, 0, 0),
        (WindowedFreqTriangleRoPEBlock, 32, 16),
        (GlobalFreqTriangleRoPEBlock, 0, 0),
    ]


@pytest.mark.parametrize("shift", [0, 3])
def test_windowed_time_block_is_local(shift: int) -> None:
    torch.manual_seed(0)
    block = WindowedTimeTriangleRoPEBlock(
        dim=32, tri_dim=16, time_window=6, window_shift=shift
    ).eval()
    x = torch.randn(1, 20, 5, 32)
    y1 = block(x)
    # Window containing frame 0 covers [0, 6 - shift); perturb everything after it.
    first_end = 6 - shift
    x_mod = x.clone()
    x_mod[:, first_end:] = torch.randn_like(x_mod[:, first_end:])
    y2 = block(x_mod)
    torch.testing.assert_close(y1[:, :first_end], y2[:, :first_end])
    assert not torch.allclose(y1[:, first_end:], y2[:, first_end:])


def test_windowed_time_matches_global_when_window_covers_chunk() -> None:
    torch.manual_seed(0)
    global_block = TimeTriangleRoPEBlock(dim=32, tri_dim=16).eval()
    windowed = WindowedTimeTriangleRoPEBlock(dim=32, tri_dim=16, time_window=10).eval()
    windowed.load_state_dict(global_block.state_dict())
    x = torch.randn(2, 10, 6, 32)
    torch.testing.assert_close(windowed(x), global_block(x), atol=1e-5, rtol=1e-5)


def test_log_frequency_positions_are_pitch_invariant() -> None:
    pos = log_frequency_positions([110.0, 220.0, 440.0, 880.0, 10.0])
    diffs = pos[1:4] - pos[0:3]
    torch.testing.assert_close(diffs, torch.full_like(diffs, 12.0))
    assert pos[4] == 0.0  # clamped to min_hz


def test_freq_block_requires_positions() -> None:
    with pytest.raises(ValueError):
        GlobalFreqTriangleRoPEBlock(dim=32, tri_dim=16)


def test_2d_rejects_band_count_mismatch() -> None:
    model = TriangleRoFormer2D(_band_freqs(8), dim=32, depth=2, tri_dim=16)
    with pytest.raises(ValueError):
        model(torch.randn(1, 8, 9, 32))


def test_2d_causal_no_future_leakage() -> None:
    torch.manual_seed(0)
    model = TriangleRoFormer2D(
        _band_freqs(8),
        dim=64,
        depth=2,
        heads=16,
        causal=True,
        use_boundary_blocks=True,
        chunk_size=4,
    ).eval()

    x = torch.randn(1, 12, 8, 64)
    y1 = model(x)

    x_mod = x.clone()
    x_mod[:, 6:, :, :] = torch.randn_like(x_mod[:, 6:, :, :])
    y2 = model(x_mod)

    torch.testing.assert_close(y1[:, :6], y2[:, :6], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(y1[:, 6:], y2[:, 6:])


def test_2d_without_boundary_blocks() -> None:
    model = TriangleRoFormer2D(
        _band_freqs(8),
        dim=64,
        depth=2,
        heads=16,
        causal=False,
        freq_window=4,
        short_freq_window=2,
        time_window=4,
        use_boundary_blocks=False,
    )
    x = torch.randn(2, 10, 8, 64)
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
