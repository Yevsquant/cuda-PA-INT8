"""Asymmetric per-token INT8 K quantization (Phase 4).

CPU tests guard the affine quant arithmetic; GPU tests check that the kernel's
S_q dot-product correction (score = (dot - z_k·Σq)·s_k·scale) matches an
asym-dequant reference, and that an all-zero zero-point reproduces the symmetric
path exactly (the gated branch is a true no-op when z=0).
"""

import math

import pytest
import torch

from paged_decode_attn import (
    build_k_cache_int8_asym,
    build_paged_kv_cache_int8,
    dequantize_kv,
    paged_decode_attention_reference,
    quantize_per_token,
    quantize_per_token_asym,
)
from quant_metrics import rel_l2
from test_paged_decode_attn import HEAD_DIM, BLOCK_SIZE, X, _make_lens

INT8_VARIANTS = ["warp_int8", "splitk_int8"]


# --- CPU: affine quant arithmetic --------------------------------------------

def test_asym_roundtrip_bound():
    torch.manual_seed(0)
    x = torch.randn(8, 256)
    q, s, z = quantize_per_token_asym(x, dim=-1)
    assert q.dtype == torch.int8 and int(q.min()) >= -128 and int(q.max()) <= 127
    dq = (q.float() - z.unsqueeze(-1)) * s.unsqueeze(-1)
    assert (dq - x).norm() / x.norm() < 0.02


def test_asym_beats_symmetric_on_skewed():
    """On a one-sided (all-positive, skewed) distribution, symmetric quant wastes
    the negative half of the range; asymmetric does not."""
    torch.manual_seed(0)
    x = torch.distributions.Exponential(1.0).sample((256, 128))

    qs, ss = quantize_per_token(x, dim=-1)
    sym = rel_l2(qs.float() * ss.unsqueeze(-1), x, dim=-1).mean()
    qa, sa, za = quantize_per_token_asym(x, dim=-1)
    asym = rel_l2((qa.float() - za.unsqueeze(-1)) * sa.unsqueeze(-1), x, dim=-1).mean()

    assert asym < 0.7 * sym, (asym.item(), sym.item())


# --- GPU: kernel correctness -------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("variant", INT8_VARIANTS)
@pytest.mark.parametrize("clen", [128, 600])  # 600 exercises split-K multi-partition
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 8), (8, 2)])
def test_int8_asym_kernel_matches_reference(variant, clen, num_heads, num_kv_heads):
    from cuda_ext import VARIANTS_INT8
    fn = VARIANTS_INT8[variant]

    torch.manual_seed(0)
    device = "cuda"
    num_seqs = 2
    lens = _make_lens(clen, num_seqs, device)
    max_len = int(lens.max())
    scale = 1.0 / math.sqrt(HEAD_DIM)

    q = torch.randn(num_seqs, num_heads, HEAD_DIM, dtype=torch.float16, device=device)
    # Skewed K (shifted off zero) so zero-points are non-trivial; V normal.
    shape = (num_seqs, max_len, num_kv_heads, HEAD_DIM)
    k = (torch.randn(shape, device=device) * 0.5 + 2.0).half()
    v = torch.randn(shape, device=device).half()

    kc_s, vc, _, vs, bt = build_paged_kv_cache_int8(
        k, v, lens, block_size=BLOCK_SIZE, x=X, mode="per_token")
    kc_a, ks_a, kz = build_k_cache_int8_asym(
        k, lens, bt, kc_s.shape[0], block_size=BLOCK_SIZE, x=X)

    k_deq, v_deq = dequantize_kv(kc_a, vc, ks_a, vs, k_zeros=kz)
    ref = paged_decode_attention_reference(
        q.float(), k_deq, v_deq, bt, lens, block_size=BLOCK_SIZE)

    out = torch.empty_like(q)
    fn(out, q, kc_a, vc, ks_a, vs, bt, lens, scale, BLOCK_SIZE, k_zeros=kz)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("variant", INT8_VARIANTS)
def test_asym_zero_points_zero_equals_symmetric(variant):
    """All-zero zero-points must reproduce the symmetric kernel output bit-for-bit
    — the correction term is z_k·Σq == 0, so the gated path is a true no-op."""
    from cuda_ext import VARIANTS_INT8
    fn = VARIANTS_INT8[variant]

    torch.manual_seed(0)
    device = "cuda"
    num_seqs = 2
    lens = _make_lens(300, num_seqs, device)
    max_len = int(lens.max())
    scale = 1.0 / math.sqrt(HEAD_DIM)

    q = torch.randn(num_seqs, 8, HEAD_DIM, dtype=torch.float16, device=device)
    shape = (num_seqs, max_len, 2, HEAD_DIM)
    k = torch.randn(shape, device=device).half()
    v = torch.randn(shape, device=device).half()
    kc, vc, ks, vs, bt = build_paged_kv_cache_int8(
        k, v, lens, block_size=BLOCK_SIZE, x=X, mode="per_token")

    out_sym = torch.empty_like(q)
    fn(out_sym, q, kc, vc, ks, vs, bt, lens, scale, BLOCK_SIZE)
    out_z0 = torch.empty_like(q)
    fn(out_z0, q, kc, vc, ks, vs, bt, lens, scale, BLOCK_SIZE, k_zeros=torch.zeros_like(ks))

    torch.testing.assert_close(out_z0, out_sym, atol=0, rtol=0)
