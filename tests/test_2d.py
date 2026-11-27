from __future__ import annotations

import torch

from triangle_roformer import TriangleRoFormer2D


def test_2d_non_causal_forward_backward() -> None:
    model = TriangleRoFormer2D(
        dim=64,
        depth=2,
        tri_dim=32,
        causal=False,
        freq_window=8,
        use_boundary_blocks=True,
    )
    x = torch.randn(1, 16, 12, 64, requires_grad=True)
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_2d_causal_no_future_leakage() -> None:
    torch.manual_seed(0)
    model = TriangleRoFormer2D(
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
        dim=64,
        depth=2,
        heads=16,
        causal=False,
        freq_window=4,
        use_boundary_blocks=False,
    )
    x = torch.randn(2, 10, 8, 64)
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
