"""Unit tests for the INT8 quantization helpers in paged_decode_attn.

These guard the quant arithmetic directly (scale formula, int8 range, round-trip
error, dequant broadcasting) — until now they were only exercised indirectly
through the CUDA kernels. Pure-torch, so they run on CPU without a GPU.
"""

import torch

from paged_decode_attn import (
    quantize_per_token,
    quantize_per_tensor,
    dequantize_kv,
)


def test_per_token_scale_formula_and_range():
    torch.manual_seed(0)
    x = torch.randn(4, 7, 128)
    q, s = quantize_per_token(x, dim=-1)
    # One scale per token (head_dim reduced away); s = max(|x|)/127.
    expected = x.abs().amax(dim=-1).clamp_min(1e-8) / 127.0
    torch.testing.assert_close(s, expected)
    assert s.shape == (4, 7)
    assert q.dtype == torch.int8
    assert int(q.abs().max()) <= 127


def test_per_tensor_scale_formula_and_range():
    torch.manual_seed(0)
    x = torch.randn(4, 7, 128)
    q, s = quantize_per_tensor(x)
    expected = x.abs().amax().clamp_min(1e-8) / 127.0
    torch.testing.assert_close(s, expected)
    assert s.ndim == 0  # one scalar for the whole tensor
    assert q.dtype == torch.int8
    assert int(q.abs().max()) <= 127


def test_per_token_roundtrip_bound():
    """Symmetric quant with scale = amax/127 ⇒ per-element error ≤ scale/2, so the
    norm-relative round-trip error on N(0,1) is well under 2%."""
    torch.manual_seed(0)
    x = torch.randn(8, 256)
    q, s = quantize_per_token(x, dim=-1)
    dq = q.float() * s.unsqueeze(-1)
    rel = (dq - x).norm() / x.norm()
    assert rel < 0.02, rel


def test_per_token_outlier_sets_scale():
    """A single outlier in a token vector inflates that token's scale and
    saturates at ±127; the rest of the vector still quantizes without overflow."""
    torch.manual_seed(0)
    x = torch.randn(2, 64)
    x[0, 0] = 1000.0
    q, s = quantize_per_token(x, dim=-1)
    assert int(q.abs().max()) <= 127
    assert q[0, 0].abs().item() == 127          # the outlier saturates
    # Token 0's scale is driven by the outlier; token 1's is not.
    assert s[0] > s[1]


def test_dequantize_kv_broadcast():
    """K scale broadcasts over (head_dim/x, x); V scale over head_dim — both per
    (block, kv_head, token-in-block)."""
    nb, H, D, bs, x = 2, 2, 8, 4, 2
    kc = torch.randint(-127, 128, (nb, H, D // x, bs, x), dtype=torch.int8)
    vc = torch.randint(-127, 128, (nb, H, D, bs), dtype=torch.int8)
    ks = torch.rand(nb, H, bs)
    vs = torch.rand(nb, H, bs)

    kd, vd = dequantize_kv(kc, vc, ks, vs)
    assert kd.shape == kc.shape and vd.shape == vc.shape

    # Concrete element: dequant must apply the token's scale to the int8 value.
    assert kd[1, 0, 2, 3, 1].item() == kc[1, 0, 2, 3, 1].item() * ks[1, 0, 3].item()
    assert vd[1, 1, 5, 2].item() == vc[1, 1, 5, 2].item() * vs[1, 1, 2].item()


def test_dequantize_kv_matches_builder_roundtrip():
    """End-to-end: dequant of a per-token int8 cache reproduces the original K/V
    (at the populated slots) within the quant step."""
    from paged_decode_attn import build_paged_kv_cache_int8

    torch.manual_seed(0)
    num_seqs, max_len, H, D = 2, 40, 2, 128
    block_size, x = 16, 8
    lens = torch.tensor([40, 33], dtype=torch.int32)
    k = torch.randn(num_seqs, max_len, H, D)
    v = torch.randn(num_seqs, max_len, H, D)

    kc, vc, ks, vs, bt = build_paged_kv_cache_int8(
        k, v, lens, block_size=block_size, x=x, mode="per_token")
    k_deq, v_deq = dequantize_kv(kc, vc, ks, vs)

    # Gather token 0 of seq 0 back out of the paged layout and compare.
    phys = int(bt[0, 0])
    k0 = k_deq[phys, :, :, 0, :].reshape(H, D)   # [H, D/x, x] -> [H, D]
    v0 = v_deq[phys, :, :, 0]                     # [H, D]
    # Per-token scale = amax/127, so |err| <= scale/2 per element.
    tol_k = (k[0, 0].abs().amax(dim=-1) / 127.0 / 2 + 1e-6)
    tol_v = (v[0, 0].abs().amax(dim=-1) / 127.0 / 2 + 1e-6)
    assert (k0 - k[0, 0]).abs().max(dim=-1).values.le(tol_k).all()
    assert (v0 - v[0, 0]).abs().max(dim=-1).values.le(tol_v).all()
