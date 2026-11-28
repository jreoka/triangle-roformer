from __future__ import annotations

import pytest
import torch

from triangle_roformer import (
    CausalFreqTriangleRoPEBlock,
    FreqRoPEAttentionBlock,
    GlobalFreqTriangleRoPEBlock,
    HopfieldFFN,
    TimeRoPEAttentionBlock,
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
        hopfield_patterns=32,
    )
    time_kinds = [
        (type(layer), getattr(layer, "time_window", 0), getattr(layer, "time_window_shift", 0))
        for layer in model.time_layers
    ]
    assert time_kinds == [
        (TimeTriangleRoPEBlock, 0, 0),
        (WindowedTimeTriangleRoPEBlock, 64, 0),
        (TimeTriangleRoPEBlock, 0, 0),
        (TimeRoPEAttentionBlock, 0, 0),
        (TimeTriangleRoPEBlock, 0, 0),
        (WindowedTimeTriangleRoPEBlock, 64, 32),
        (TimeTriangleRoPEBlock, 0, 0),
        (TimeRoPEAttentionBlock, 0, 0),
    ]
    freq_kinds = [
        (type(layer), getattr(layer, "freq_window", 0), getattr(layer, "window_shift", 0))
        for layer in model.freq_layers
    ]
    assert freq_kinds == [
        (WindowedFreqTriangleRoPEBlock, 8, 0),
        (GlobalFreqTriangleRoPEBlock, 0, 0),
        (WindowedFreqTriangleRoPEBlock, 32, 0),
        (FreqRoPEAttentionBlock, 0, 0),
        (WindowedFreqTriangleRoPEBlock, 8, 4),
        (GlobalFreqTriangleRoPEBlock, 0, 0),
        (WindowedFreqTriangleRoPEBlock, 32, 16),
        (FreqRoPEAttentionBlock, 0, 0),
    ]
    assert isinstance(model.freq_layers[3].hopfield_ffn, HopfieldFFN)
    assert isinstance(model.freq_layers[7].hopfield_ffn, HopfieldFFN)


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


# --------------------------------------------------------------------------
# Separation-oriented options: frame mask, soft norm, sink, causal decay
# --------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True])
def test_frame_mask_matches_unpadded_input(causal: bool) -> None:
    torch.manual_seed(0)
    model = TriangleRoFormer2D(
        _band_freqs(8),
        dim=64,
        depth=4,
        tri_dim=32,
        causal=causal,
        freq_window=4,
        short_freq_window=2,
        time_window=4,
        chunk_size=3,
    ).eval()
    x = torch.randn(2, 10, 8, 64)
    ref = model(x)

    padded = torch.cat((x, 5.0 * torch.randn(2, 5, 8, 64)), dim=1)
    mask = torch.ones(2, 15, dtype=torch.bool)
    mask[:, 10:] = False
    out = model(padded, frame_mask=mask)
    torch.testing.assert_close(out[:, :10], ref, atol=1e-5, rtol=1e-5)

    if not causal:
        # Without the mask the padding leaks into the real frames.
        assert not torch.allclose(model(padded)[:, :10], ref, atol=1e-3)


def test_frame_mask_per_sample_lengths() -> None:
    torch.manual_seed(0)
    model = TriangleRoFormer2D(
        _band_freqs(6), dim=32, depth=2, tri_dim=16,
        freq_window=4, short_freq_window=2, time_window=4,
    ).eval()
    x = torch.randn(2, 12, 6, 32)
    mask = torch.ones(2, 12)
    mask[1, 7:] = 0.0
    out = model(x, frame_mask=mask)
    torch.testing.assert_close(out[0], model(x[:1])[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[1, :7], model(x[1:, :7])[0], atol=1e-5, rtol=1e-5)


def test_frame_mask_shape_is_validated() -> None:
    model = TriangleRoFormer2D(_band_freqs(4), dim=32, depth=1, tri_dim=16)
    with pytest.raises(ValueError):
        model(torch.randn(1, 6, 4, 32), frame_mask=torch.ones(1, 5))


def test_soft_norm_keeps_weak_correlations_weak() -> None:
    soft = TimeTriangleRoPEBlock(dim=32, tri_dim=16)
    hard = TimeTriangleRoPEBlock(dim=32, tri_dim=16, soft_norm=False)
    k_r, k_i = torch.randn(1, 8, 5, 10), torch.randn(1, 8, 5, 10)
    inv_n = torch.full((1, 1, 1, 1), 1.0 / 64)

    def norm(block, scale):
        out_r, out_i = block._normalize_kernel(k_r * scale, k_i * scale, inv_n)
        return (out_r.square() + out_i.square()).mean().sqrt().item()

    # Hard normalization erases the correlation strength ...
    assert norm(hard, 1e-2) == pytest.approx(norm(hard, 1.0), rel=1e-2)
    # ... soft normalization keeps noise-level kernels small and strong ones ~unit.
    assert norm(soft, 1.0) > 0.95
    assert norm(soft, 1e-2) < 0.3  # ~1/4 of the 1/sqrt(n) noise level


def test_attention_sink_lets_rows_attend_to_nothing() -> None:
    pos = log_frequency_positions(_band_freqs(6))
    with_sink = GlobalFreqTriangleRoPEBlock(dim=32, tri_dim=16, freq_positions=pos)
    no_sink = GlobalFreqTriangleRoPEBlock(
        dim=32, tri_dim=16, freq_positions=pos, attn_sink=False
    )
    m = torch.randn(2, 8, 6, 6)
    inv_n = torch.full((1, 1, 1, 1), 0.1)
    torch.testing.assert_close(
        no_sink._freq_attention(m, inv_n).sum(-1), torch.ones(2, 8, 6)
    )
    with torch.no_grad():
        with_sink.sink_logit.fill_(10.0)
    assert (with_sink._freq_attention(m, inv_n).sum(-1) < 0.5).all()


def test_causal_freq_without_decay_ends_at_global_affinity() -> None:
    torch.manual_seed(0)
    pos = log_frequency_positions(_band_freqs(6))
    causal = CausalFreqTriangleRoPEBlock(
        dim=32, tri_dim=16, freq_positions=pos, chunk_size=4, causal_decay=False
    ).eval()
    glob = GlobalFreqTriangleRoPEBlock(dim=32, tri_dim=16, freq_positions=pos).eval()
    glob.load_state_dict(causal.state_dict())
    x = torch.randn(2, 11, 6, 32)
    # The last frame's prefix is the whole chunk.
    torch.testing.assert_close(causal(x)[:, -1], glob(x)[:, -1], atol=1e-5, rtol=1e-5)


def test_causal_freq_with_fast_decay_is_frame_local() -> None:
    torch.manual_seed(0)
    pos = log_frequency_positions(_band_freqs(6))
    causal = CausalFreqTriangleRoPEBlock(
        dim=32, tri_dim=16, freq_positions=pos, chunk_size=4
    ).eval()
    with torch.no_grad():
        causal.freq_decay_logit.fill_(-40.0)
    local = WindowedFreqTriangleRoPEBlock(
        dim=32, tri_dim=16, freq_positions=pos, freq_window=1
    ).eval()
    local.load_state_dict(causal.state_dict(), strict=False)
    x = torch.randn(2, 9, 6, 32)
    torch.testing.assert_close(causal(x), local(x), atol=1e-5, rtol=1e-5)


def test_causal_time_without_decay_ends_at_global_kernel() -> None:
    torch.manual_seed(0)
    causal = TimeTriangleRoPEBlock(
        dim=32, tri_dim=16, causal=True, chunk_size=4, causal_decay=False
    ).eval()
    glob = TimeTriangleRoPEBlock(dim=32, tri_dim=16).eval()
    glob.load_state_dict(causal.state_dict())
    x = torch.randn(2, 11, 6, 32)
    torch.testing.assert_close(causal(x)[:, -1], glob(x)[:, -1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("block_cls", ["time", "freq"])
def test_causal_decay_is_chunk_size_invariant(block_cls: str) -> None:
    torch.manual_seed(0)
    pos = log_frequency_positions(_band_freqs(6))

    def build(chunk: int):
        torch.manual_seed(1)
        if block_cls == "time":
            return TimeTriangleRoPEBlock(dim=32, tri_dim=16, causal=True, chunk_size=chunk)
        return CausalFreqTriangleRoPEBlock(
            dim=32, tri_dim=16, freq_positions=pos, chunk_size=chunk
        )

    x = torch.randn(2, 13, 6, 32)
    torch.testing.assert_close(build(3).eval()(x), build(64).eval()(x), atol=1e-5, rtol=1e-5)


def test_causal_decay_reduces_old_frame_influence() -> None:
    torch.manual_seed(0)
    block = TimeTriangleRoPEBlock(dim=32, tri_dim=16, causal=True, chunk_size=8).eval()
    with torch.no_grad():
        block.time_decay_logit.fill_(0.0)  # gamma = 0.5 for every channel
    x = torch.randn(1, 40, 6, 32)
    x_mod = x.clone()
    x_mod[:, :5] = torch.randn_like(x_mod[:, :5])
    diff = (block(x) - block(x_mod)).abs().amax(dim=(0, 2, 3))
    assert diff[-1] < 1e-4 * diff[5]


def test_band_embedding_is_zero_initialized() -> None:
    torch.manual_seed(0)
    base = TriangleRoFormer2D(_band_freqs(6), dim=32, depth=2, tri_dim=16).eval()
    emb = TriangleRoFormer2D(
        _band_freqs(6), dim=32, depth=2, tri_dim=16, band_embedding=True
    ).eval()
    emb.load_state_dict(base.state_dict(), strict=False)
    assert emb.band_emb is not None and emb.band_emb.shape == (6, 32)
    x = torch.randn(1, 8, 6, 32)
    torch.testing.assert_close(emb(x), base(x))


@pytest.mark.parametrize("causal", [False, True])
def test_new_parameters_receive_gradients(causal: bool) -> None:
    torch.manual_seed(0)
    model = TriangleRoFormer2D(
        _band_freqs(6), dim=32, depth=2, tri_dim=16, causal=causal,
        freq_window=4, short_freq_window=2, time_window=4, chunk_size=4,
        band_embedding=True,
    )
    model(torch.randn(2, 9, 6, 32)).square().mean().backward()
    names = ("log_norm_floor", "sink_logit", "decay_logit", "band_emb")
    tracked = [(n, p) for n, p in model.named_parameters() if any(k in n for k in names)]
    assert any("decay_logit" in n for n, _ in tracked) == causal
    for name, param in tracked:
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


# --------------------------------------------------------------------------
# Hybrid RoPE attention + HopfieldFFN tests
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_steps", [1, 2, 3])
def test_hopfield_ffn_shapes_and_gradients(num_steps: int) -> None:
    torch.manual_seed(0)
    ffn = HopfieldFFN(
        dim=64, num_heads=4, num_patterns=32, num_steps=num_steps, attn_sink=True
    )
    x = torch.randn(2, 6, 8, 64, requires_grad=True)
    y = ffn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in ffn.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_hopfield_ffn_sink_suppresses_output() -> None:
    torch.manual_seed(0)
    ffn = HopfieldFFN(dim=32, num_heads=2, num_patterns=16, num_steps=2, attn_sink=True).eval()
    x = torch.randn(1, 4, 6, 32)
    y_normal = ffn(x)
    with torch.no_grad():
        ffn.sink_logit.fill_(25.0)
    y_sunk = ffn(x)
    assert y_sunk.abs().max() < 1e-4 * y_normal.abs().max()


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("attn_sink", [False, True])
def test_rope_attention_blocks_frame_mask_and_causality(
    causal: bool, attn_sink: bool
) -> None:
    torch.manual_seed(0)
    model = TriangleRoFormer2D(
        _band_freqs(6),
        dim=32,
        depth=4,
        tri_dim=16,
        causal=causal,
        attn_sink=attn_sink,
        hopfield_patterns=32,
        hopfield_steps=2,
        freq_window=4,
        short_freq_window=2,
        time_window=4,
        chunk_size=3,
    ).eval()

    x = torch.randn(2, 10, 6, 32)
    ref = model(x)

    # Frame mask matches unpadded input
    padded = torch.cat((x, 5.0 * torch.randn(2, 4, 6, 32)), dim=1)
    mask = torch.ones(2, 14, dtype=torch.bool)
    mask[:, 10:] = False
    out = model(padded, frame_mask=mask)
    torch.testing.assert_close(out[:, :10], ref, atol=1e-5, rtol=1e-5)

    # Causal mode has zero future leakage
    if causal:
        x_mod = x.clone()
        x_mod[:, 5:] = torch.randn_like(x_mod[:, 5:])
        y_mod = model(x_mod)
        torch.testing.assert_close(ref[:, :5], y_mod[:, :5], atol=1e-5, rtol=1e-5)
        assert not torch.allclose(ref[:, 5:], y_mod[:, 5:])


@pytest.mark.parametrize("causal", [False, True])
def test_hybrid_parameters_receive_gradients(causal: bool) -> None:
    torch.manual_seed(0)
    model = TriangleRoFormer2D(
        _band_freqs(6),
        dim=32,
        depth=4,
        tri_dim=16,
        causal=causal,
        hopfield_patterns=32,
        hopfield_steps=2,
        freq_window=4,
        short_freq_window=2,
        time_window=4,
        chunk_size=3,
    )
    model(torch.randn(2, 9, 6, 32)).square().mean().backward()
    names = ("sink_logit", "hopfield_ffn.keys", "hopfield_ffn.values", "hopfield_ffn.log_beta")
    tracked = [(n, p) for n, p in model.named_parameters() if any(k in n for k in names)]
    assert len(tracked) > 0
    for name, param in tracked:
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_checkpoint_works_when_input_has_no_grad() -> None:
    torch.manual_seed(0)
    model = TriangleRoFormer2D(
        _band_freqs(6),
        dim=32,
        depth=4,
        tri_dim=16,
        use_checkpoint=True,
        drop_path=0.0,
        hopfield_patterns=16,
    )
    x = torch.randn(1, 8, 6, 32)  # requires_grad=False
    y = model(x)
    assert y.requires_grad
    y.square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("attn_sink", [False, True])
@pytest.mark.parametrize("step_size", [0.5, 1.0])
def test_hopfield_ffn_matches_explicit_attention(
    attn_sink: bool, step_size: float
) -> None:
    torch.manual_seed(0)
    ffn = HopfieldFFN(
        dim=48,
        num_heads=3,
        num_patterns=20,
        num_steps=2,
        step_size=step_size,
        attn_sink=attn_sink,
    ).eval()
    x = torch.randn(2, 5, 4, 48)
    y = ffn(x)

    # Explicit reference without SDPA
    xi_raw, out_gate = ffn.in_proj(x).chunk(2, dim=-1)
    xi = xi_raw.reshape(-1, ffn.num_heads, ffn.head_dim).transpose(0, 1)
    xi = torch.nn.functional.normalize(xi.float(), dim=-1, eps=1e-6)
    keys = torch.nn.functional.normalize(ffn.keys.float(), dim=-1, eps=1e-6)
    beta = ffn.log_beta.float().clamp(torch.log(torch.tensor(0.5)), torch.log(torch.tensor(32.0))).exp().view(-1, 1, 1)
    mass: torch.Tensor | float = 1.0
    for step in range(ffn.num_steps):
        if step == 0:
            xi_n = xi
        else:
            xi_u = torch.nn.functional.normalize(xi, dim=-1, eps=1e-6)
            xi_n = xi_u * mass if attn_sink else xi_u
        logits = torch.matmul(xi_n, keys.transpose(-2, -1)) * beta
        if attn_sink:
            sink = ffn.sink_logit.float().view(-1, 1, 1).expand(logits.shape[0], logits.shape[1], 1)
            weights = torch.softmax(torch.cat((logits, sink), dim=-1), dim=-1)[..., :-1]
        else:
            weights = torch.softmax(logits, dim=-1)
        if step < ffn.num_steps - 1:
            if attn_sink:
                p_real = weights.sum(dim=-1, keepdim=True)
                mass = p_real if step_size == 1.0 else mass + step_size * (p_real - mass)
            xi = xi_n + step_size * (torch.matmul(weights, keys) - xi_n)
    out = torch.matmul(weights, ffn.values.float()).transpose(0, 1).reshape_as(x)
    ref = ffn.out_proj(out) * torch.sigmoid(out_gate)
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("causal_decay", [False, True])
def test_causal_time_matches_frame_by_frame_reference(causal_decay: bool) -> None:
    torch.manual_seed(0)
    block = TimeTriangleRoPEBlock(
        dim=32, tri_dim=16, causal=True, chunk_size=3, causal_decay=causal_decay
    ).eval()
    x = torch.randn(2, 9, 5, 32)
    mask = torch.ones(2, 9)
    mask[1, 6:] = 0.0
    # chunk_size=1 runs pure frame-by-frame recurrence; compare against chunk_size=3 and chunk_size=16
    block_1 = TimeTriangleRoPEBlock(
        dim=32, tri_dim=16, causal=True, chunk_size=1, causal_decay=causal_decay
    ).eval()
    block_1.load_state_dict(block.state_dict())
    block_16 = TimeTriangleRoPEBlock(
        dim=32, tri_dim=16, causal=True, chunk_size=16, causal_decay=causal_decay
    ).eval()
    block_16.load_state_dict(block.state_dict())
    y3 = block(x, frame_mask=mask)
    y1 = block_1(x, frame_mask=mask)
    y16 = block_16(x, frame_mask=mask)
    torch.testing.assert_close(y3, y1, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(y3, y16, atol=1e-5, rtol=1e-5)


def test_rope_attention_sink_matches_explicit_scalar_logit() -> None:
    torch.manual_seed(0)
    pos = log_frequency_positions(_band_freqs(6))
    x = torch.randn(2, 8, 6, 32)

    for block in (
        TimeRoPEAttentionBlock(dim=32, heads=2, attn_sink=True).eval(),
        FreqRoPEAttentionBlock(dim=32, heads=2, freq_positions=pos, attn_sink=True).eval(),
    ):
        with torch.no_grad():
            block.sink_logit.copy_(torch.tensor([1.5, -0.75]))
        y = block(x)

        # Explicit reference with position-invariant scalar sink_logit
        batch, frames, bands, dim = x.shape
        if block.axis == "time":
            h = block.attn_norm(x).permute(0, 2, 1, 3).reshape(batch * bands, frames, dim)
            seq_len = frames
        else:
            h = block.attn_norm(x).reshape(batch * frames, bands, dim)
            seq_len = bands
        q, k, v = block.qkv_proj(h).chunk(3, dim=-1)
        q = block.q_norm(q.reshape(-1, seq_len, block.heads, block.head_dim)).transpose(1, 2)
        k = block.k_norm(k.reshape(-1, seq_len, block.heads, block.head_dim)).transpose(1, 2)
        v = v.reshape(-1, seq_len, block.heads, block.head_dim).transpose(1, 2)
        if block.axis == "time":
            q, k = block.rope.apply_qk(q, k)
        else:
            q, k = block.rope.apply_qk_tables(q, k, block.freq_cos, block.freq_sin)
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (block.head_dim ** -0.5)
        sink = block.sink_logit.float().view(1, block.heads, 1, 1).expand(logits.shape[0], -1, seq_len, 1)
        weights = torch.softmax(torch.cat((sink, logits), dim=-1), dim=-1)[..., 1:]
        o = torch.matmul(weights, v.float()).transpose(1, 2)
        if block.axis == "time":
            o = block.out_proj(o.reshape(batch, bands, frames, dim)).permute(0, 2, 1, 3)
            x_mid = x + o
            gate, val = block.ffn_in(block.ffn_norm(x_mid)).chunk(2, dim=-1)
            ref = x_mid + block.ffn_out(torch.nn.functional.silu(gate) * val)
        else:
            o = block.out_proj(o.reshape(batch, frames, bands, dim))
            x_mid = x + o
            ref = x_mid + block.hopfield_ffn(block.ffn_norm(x_mid))
        torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)

        # Large sink_logit suppresses the attention residual cleanly
        with torch.no_grad():
            block.sink_logit.fill_(25.0)
        y_sunk = block(x)
        if block.axis == "time":
            gate, val = block.ffn_in(block.ffn_norm(x)).chunk(2, dim=-1)
            ref_no_attn = x + block.ffn_out(torch.nn.functional.silu(gate) * val)
        else:
            ref_no_attn = x + block.hopfield_ffn(block.ffn_norm(x))
        torch.testing.assert_close(y_sunk, ref_no_attn, atol=1e-5, rtol=1e-5)


def test_hopfield_ffn_multistep_sink_preserves_shrinkage() -> None:
    torch.manual_seed(0)
    ffn = HopfieldFFN(
        dim=32, num_heads=2, num_patterns=16, num_steps=2, step_size=1.0, attn_sink=True
    ).eval()
    # Set moderate sink_logit so step 0 puts most mass on sink (but not saturating fp32):
    # with mass preserved, step 1 logits shrink toward 0 instead of snapping back to beta (~4.0).
    with torch.no_grad():
        ffn.sink_logit.fill_(5.0)
    x = torch.randn(1, 4, 6, 32)
    y_2step = ffn(x)

    ffn_1step = HopfieldFFN(
        dim=32, num_heads=2, num_patterns=16, num_steps=1, step_size=1.0, attn_sink=True
    ).eval()
    ffn_1step.load_state_dict(ffn.state_dict())
    y_1step = ffn_1step(x)
    # Step 2 should stay at least as suppressed as step 1 rather than rebounding.
    assert y_2step.abs().mean() <= y_1step.abs().mean() * 1.1


def test_scalar_control_params_are_1d_for_no_weight_decay() -> None:
    model = TriangleRoFormer2D(
        _band_freqs(6),
        dim=32,
        depth=4,
        tri_dim=16,
        hopfield_patterns=16,
        freq_window=4,
        short_freq_window=2,
        time_window=4,
    )
    control_keys = ("sink_logit", "log_beta", "freq_scale", "log_norm_floor", "decay_logit")
    matched = [
        (name, param)
        for name, param in model.named_parameters()
        if any(k in name for k in control_keys)
    ]
    assert len(matched) > 0
    for name, param in matched:
        assert param.ndim == 1, f"{name} has shape {tuple(param.shape)}, expected 1D"


def test_moderate_sink_suppresses_hopfield_and_freq_triangle_without_rms_blowup() -> None:
    torch.manual_seed(0)
    # Realistic sink_logit (~10.0, well above eps=1e-6 so RMSNorm would have cancelled it)
    ffn = HopfieldFFN(dim=32, num_heads=2, num_patterns=16, num_steps=2, attn_sink=True).eval()
    x = torch.randn(1, 4, 6, 32)
    y_normal = ffn(x)
    with torch.no_grad():
        ffn.sink_logit.fill_(10.0)
    y_sunk = ffn(x)
    assert y_sunk.abs().mean() < 0.05 * y_normal.abs().mean()

    pos = log_frequency_positions(_band_freqs(6))
    freq_blk = GlobalFreqTriangleRoPEBlock(dim=32, tri_dim=16, freq_positions=pos, attn_sink=True).eval()
    with torch.no_grad():
        # Isolate the triangle residual from FFN by zeroing ffn_out
        freq_blk.ffn_out.weight.zero_()
    delta_normal = freq_blk(x) - x
    with torch.no_grad():
        freq_blk.sink_logit.fill_(10.0)
    delta_sunk = freq_blk(x) - x
    assert delta_sunk.abs().mean() < 0.05 * delta_normal.abs().mean()
