from __future__ import annotations

import torch

from triangle_roformer import TriangleRoFormerBlock1D, TriangleRoFormerLM, TriangleRoFormer


def test_1d_non_causal_forward_backward() -> None:
    model = TriangleRoFormer(dim=64, depth=3, heads=4, causal=False)
    x = torch.randn(2, 20, 64, requires_grad=True)
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_1d_causal_no_future_leakage() -> None:
    torch.manual_seed(0)
    model = TriangleRoFormer(dim=64, depth=3, heads=4, causal=True, chunk_size=8)
    model.eval()

    x = torch.randn(2, 20, 64)
    y1 = model(x)

    x_mod = x.clone()
    x_mod[:, 10:, :] = torch.randn_like(x_mod[:, 10:, :])
    y2 = model(x_mod)

    torch.testing.assert_close(y1[:, :10], y2[:, :10], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(y1[:, 10:], y2[:, 10:])


def test_1d_causal_recurrent_matches_parallel() -> None:
    torch.manual_seed(42)
    model = TriangleRoFormer(dim=64, depth=2, heads=4, causal=True, chunk_size=5)
    model.eval()

    x = torch.randn(2, 13, 64)
    y_parallel, _ = model(x, return_states=True)

    states = None
    step_outputs = []
    for t in range(x.shape[1]):
        y_t, states = model(x[:, t : t + 1], states=states, return_states=True)
        step_outputs.append(y_t)
    y_recurrent = torch.cat(step_outputs, dim=1)

    torch.testing.assert_close(y_parallel, y_recurrent, atol=1e-5, rtol=1e-5)


def test_1d_causal_matches_non_causal_at_last_token() -> None:
    torch.manual_seed(7)
    block_nc = TriangleRoFormerBlock1D(dim=64, heads=4, causal=False).eval()
    block_c = TriangleRoFormerBlock1D(dim=64, heads=4, causal=True, chunk_size=4).eval()
    block_c.load_state_dict(block_nc.state_dict())

    x = torch.randn(2, 11, 64)
    y_nc = block_nc(x)
    y_c = block_c(x)
    torch.testing.assert_close(y_nc[:, -1], y_c[:, -1], atol=1e-5, rtol=1e-5)


def test_1d_attention_mask() -> None:
    torch.manual_seed(1)
    model = TriangleRoFormer(dim=64, depth=2, heads=4, causal=False).eval()
    x = torch.randn(2, 12, 64)
    mask = torch.tensor([[1] * 8 + [0] * 4, [1] * 12], dtype=torch.bool)
    y1 = model(x, attention_mask=mask)

    x_mod = x.clone()
    x_mod[0, 8:, :] = 999.0
    y2 = model(x_mod, attention_mask=mask)
    torch.testing.assert_close(y1[0, :8], y2[0, :8], atol=1e-5, rtol=1e-5)


def test_lm_forward_and_generation_state() -> None:
    lm = TriangleRoFormerLM(vocab_size=128, dim=64, depth=2, heads=4, causal=True).eval()
    ids = torch.randint(0, 128, (2, 10))
    logits_full = lm(ids)
    assert logits_full.shape == (2, 10, 128)

    logits_prefill, states = lm(ids[:, :6], return_states=True)
    logits_rest, _ = lm(ids[:, 6:], states=states, return_states=True)
    torch.testing.assert_close(
        logits_full, torch.cat([logits_prefill, logits_rest], dim=1), atol=1e-5, rtol=1e-5
    )
