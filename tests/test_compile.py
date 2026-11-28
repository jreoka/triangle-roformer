from __future__ import annotations

import torch

from triangle_roformer import TriangleRoFormer, TriangleRoFormer2D


def _backward_with_thread_change(model: torch.nn.Module, x: torch.Tensor) -> None:
    """On CUDA, checkpoint recomputation runs on autograd's device thread, whose
    `num_threads` differs from the forward thread. Changing it between forward and
    backward reproduces that; with a recompile limit of 1, a recompute that
    re-entered Dynamo would fall back to eager and raise CheckpointError."""
    import torch._dynamo.config as dynamo_config

    threads = torch.get_num_threads()
    limit = dynamo_config.recompile_limit
    try:
        y = model(x)
        dynamo_config.recompile_limit = 1
        torch.set_num_threads(max(1, threads // 2) if threads > 1 else 2)
        y.square().mean().backward()
    finally:
        torch.set_num_threads(threads)
        dynamo_config.recompile_limit = limit
        torch._dynamo.reset()


def test_2d_compiled_checkpoint_backward_does_not_reenter_dynamo() -> None:
    torch.manual_seed(0)
    model = TriangleRoFormer2D(
        [50.0 * 1.3**i for i in range(6)], dim=32, depth=2, tri_dim=16,
        freq_window=4, short_freq_window=2, time_window=4, drop_path=0.0,
        use_checkpoint=True,
    ).train()
    model.compile_layers()
    x = torch.randn(1, 10, 6, 32, requires_grad=True)
    _backward_with_thread_change(model, x)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_1d_compiled_checkpoint_backward_does_not_reenter_dynamo() -> None:
    torch.manual_seed(0)
    model = TriangleRoFormer(dim=32, depth=2, heads=4, use_checkpoint=True).train()
    model.compile_layers()
    x = torch.randn(2, 12, 32, requires_grad=True)
    _backward_with_thread_change(model, x)
    assert x.grad is not None and torch.isfinite(x.grad).all()
