import math

import pytest
import torch

from paged_decode_attn import (
    paged_decode_attention_reference,
    build_paged_kv_cache,
    build_paged_kv_cache_int8,
    dequantize_kv,
)

HEAD_DIM = 128
BLOCK_SIZE = 16
X = 8

# context lengths that cover: single token, full block, just-over-a-block,
# partial last block, and a long sequence.
CONTEXT_LENS = [1, 15, 16, 17, 31, 128, 500]
# (num_heads, num_kv_heads): MHA, GQA-4x, MQA, GQA-4x with more kv heads.
GQA = [(8, 8), (8, 2), (8, 1), (16, 4)]
NUM_SEQS = [1, 3]


def _make_lens(clen, num_seqs, device):
    """Per-seq context lengths (varied within a batch), >= 1."""
    lens = [max(1, clen - 5 * b) for b in range(num_seqs)]
    return torch.tensor(lens, dtype=torch.int32, device=device)


def _rand_kv(num_seqs, max_len, num_kv_heads, dtype, device):
    shape = (num_seqs, max_len, num_kv_heads, HEAD_DIM)
    return (torch.randn(shape, dtype=dtype, device=device),
            torch.randn(shape, dtype=dtype, device=device))


def _dense_attention(q, k, v, context_lens, num_kv_heads):
    """Independent oracle: per-head softmax attention on contiguous K/V."""
    num_seqs, num_heads, head_dim = q.shape
    qpkv = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    out = torch.zeros_like(q)
    for b in range(num_seqs):
        clen = int(context_lens[b])
        for h in range(num_heads):
            kvh = h // qpkv
            kk = k[b, :clen, kvh, :]            # [clen, D]
            vv = v[b, :clen, kvh, :]            # [clen, D]
            scores = (kk @ q[b, h]) * scale     # [clen]
            probs = torch.softmax(scores, dim=-1)
            out[b, h] = (probs.unsqueeze(-1) * vv).sum(dim=0)
    return out


@pytest.mark.parametrize("clen", CONTEXT_LENS)
@pytest.mark.parametrize("num_heads,num_kv_heads", GQA)
@pytest.mark.parametrize("num_seqs", NUM_SEQS)
def test_reference_matches_dense(clen, num_heads, num_kv_heads, num_seqs):
    """The paged reference oracle must match a plain dense attention."""
    torch.manual_seed(0)
    device = "cpu"
    lens = _make_lens(clen, num_seqs, device)
    max_len = int(lens.max())

    q = torch.randn(num_seqs, num_heads, HEAD_DIM, dtype=torch.float32, device=device)
    k, v = _rand_kv(num_seqs, max_len, num_kv_heads, torch.float32, device)

    k_cache, v_cache, block_table = build_paged_kv_cache(
        k, v, lens, block_size=BLOCK_SIZE, x=X
    )

    ref = paged_decode_attention_reference(
        q, k_cache, v_cache, block_table, lens, block_size=BLOCK_SIZE
    )
    dense = _dense_attention(q, k, v, lens, num_kv_heads)

    torch.testing.assert_close(ref, dense, atol=1e-5, rtol=1e-5)


def _variant_names():
    from cuda_ext import VARIANTS
    return list(VARIANTS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("variant", _variant_names() if torch.cuda.is_available() else [])
@pytest.mark.parametrize("clen", CONTEXT_LENS)
@pytest.mark.parametrize("num_heads,num_kv_heads", GQA)
@pytest.mark.parametrize("num_seqs", NUM_SEQS)
def test_cuda_matches_reference(variant, clen, num_heads, num_kv_heads, num_seqs):
    """Every CUDA variant must match the reference within fp16 tolerance."""
    from cuda_ext import VARIANTS  # JIT-compiles on first call

    paged_decode_attention = VARIANTS[variant]

    torch.manual_seed(0)
    device = "cuda"
    lens = _make_lens(clen, num_seqs, device)
    max_len = int(lens.max())
    scale = 1.0 / math.sqrt(HEAD_DIM)

    q = torch.randn(num_seqs, num_heads, HEAD_DIM, dtype=torch.float16, device=device)
    k, v = _rand_kv(num_seqs, max_len, num_kv_heads, torch.float16, device)

    k_cache, v_cache, block_table = build_paged_kv_cache(
        k, v, lens, block_size=BLOCK_SIZE, x=X
    )

    # Reference in fp32 over the exact fp16-rounded cache values.
    ref = paged_decode_attention_reference(
        q.float(), k_cache.float(), v_cache.float(), block_table, lens, block_size=BLOCK_SIZE
    )

    out = torch.empty_like(q)
    paged_decode_attention(
        out, q, k_cache, v_cache, block_table, lens, scale, BLOCK_SIZE
    )

    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


# Contexts that cross split-K partition boundaries (PARTITION_SIZE=512): exactly
# on a boundary, just over it, and spanning several partitions. The clen<=500
# grid above only ever yields num_splits=1, so the multi-split merge is untested
# without these.
LONG_CONTEXT_LENS = [512, 513, 1024, 2000, 4096]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("variant", _variant_names() if torch.cuda.is_available() else [])
@pytest.mark.parametrize("clen", LONG_CONTEXT_LENS)
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 8), (8, 2)])
def test_cuda_matches_reference_long(variant, clen, num_heads, num_kv_heads):
    """Long contexts that exercise split-K's multi-partition merge path."""
    from cuda_ext import VARIANTS

    paged_decode_attention = VARIANTS[variant]

    torch.manual_seed(0)
    device = "cuda"
    num_seqs = 2
    lens = _make_lens(clen, num_seqs, device)
    max_len = int(lens.max())
    scale = 1.0 / math.sqrt(HEAD_DIM)

    q = torch.randn(num_seqs, num_heads, HEAD_DIM, dtype=torch.float16, device=device)
    k, v = _rand_kv(num_seqs, max_len, num_kv_heads, torch.float16, device)

    k_cache, v_cache, block_table = build_paged_kv_cache(
        k, v, lens, block_size=BLOCK_SIZE, x=X
    )

    ref = paged_decode_attention_reference(
        q.float(), k_cache.float(), v_cache.float(), block_table, lens, block_size=BLOCK_SIZE
    )

    out = torch.empty_like(q)
    paged_decode_attention(
        out, q, k_cache, v_cache, block_table, lens, scale, BLOCK_SIZE
    )

    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


# --- INT8 KV-cache tests ----------------------------------------------------

def _int8_variant_names():
    from cuda_ext import VARIANTS_INT8
    return list(VARIANTS_INT8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("variant", _int8_variant_names() if torch.cuda.is_available() else [])
@pytest.mark.parametrize("mode", ["per_token", "per_tensor"])
@pytest.mark.parametrize("clen", CONTEXT_LENS)
@pytest.mark.parametrize("num_heads,num_kv_heads", GQA)
@pytest.mark.parametrize("num_seqs", NUM_SEQS)
def test_int8_kernel_matches_dequant_reference(
        variant, mode, clen, num_heads, num_kv_heads, num_seqs):
    """The INT8 kernel must match the reference run on the *dequantized* cache.

    Comparing against the dequant cache (not the original fp16) isolates the
    kernel's dequant arithmetic from the quantization error itself.
    """
    from cuda_ext import VARIANTS_INT8

    paged_decode_attention = VARIANTS_INT8[variant]

    torch.manual_seed(0)
    device = "cuda"
    lens = _make_lens(clen, num_seqs, device)
    max_len = int(lens.max())
    scale = 1.0 / math.sqrt(HEAD_DIM)

    q = torch.randn(num_seqs, num_heads, HEAD_DIM, dtype=torch.float16, device=device)
    k, v = _rand_kv(num_seqs, max_len, num_kv_heads, torch.float16, device)

    k_cache, v_cache, k_scales, v_scales, block_table = build_paged_kv_cache_int8(
        k, v, lens, block_size=BLOCK_SIZE, x=X, mode=mode
    )

    # Dequant-reference oracle: rebuild fp32 K/V from int8+scales, run the
    # existing reference attention math on it.
    k_deq, v_deq = dequantize_kv(k_cache, v_cache, k_scales, v_scales)
    ref = paged_decode_attention_reference(
        q.float(), k_deq, v_deq, block_table, lens, block_size=BLOCK_SIZE
    )

    out = torch.empty_like(q)
    paged_decode_attention(
        out, q, k_cache, v_cache, k_scales, v_scales, block_table, lens, scale, BLOCK_SIZE
    )

    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("variant", ["splitk_int8"] if torch.cuda.is_available() else [])
@pytest.mark.parametrize("mode", ["per_token", "per_tensor"])
@pytest.mark.parametrize("clen", LONG_CONTEXT_LENS)
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 8), (8, 2)])
def test_int8_kernel_matches_dequant_reference_long(
        variant, mode, clen, num_heads, num_kv_heads):
    """Long contexts exercising split-K's multi-partition merge with INT8."""
    from cuda_ext import VARIANTS_INT8

    paged_decode_attention = VARIANTS_INT8[variant]

    torch.manual_seed(0)
    device = "cuda"
    num_seqs = 2
    lens = _make_lens(clen, num_seqs, device)
    max_len = int(lens.max())
    scale = 1.0 / math.sqrt(HEAD_DIM)

    q = torch.randn(num_seqs, num_heads, HEAD_DIM, dtype=torch.float16, device=device)
    k, v = _rand_kv(num_seqs, max_len, num_kv_heads, torch.float16, device)

    k_cache, v_cache, k_scales, v_scales, block_table = build_paged_kv_cache_int8(
        k, v, lens, block_size=BLOCK_SIZE, x=X, mode=mode
    )

    k_deq, v_deq = dequantize_kv(k_cache, v_cache, k_scales, v_scales)
    ref = paged_decode_attention_reference(
        q.float(), k_deq, v_deq, block_table, lens, block_size=BLOCK_SIZE
    )

    out = torch.empty_like(q)
    paged_decode_attention(
        out, q, k_cache, v_cache, k_scales, v_scales, block_table, lens, scale, BLOCK_SIZE
    )

    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("variant", _int8_variant_names() if torch.cuda.is_available() else [])
@pytest.mark.parametrize("clen", [128, 500])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 8), (8, 2)])
def test_int8_precision_metric(variant, clen, num_heads, num_kv_heads):
    """Measure (not hard-assert) INT8-vs-FP16 output error. Per-token quant
    should be no worse than per-tensor (the ablation point), and within a loose
    sanity bound."""
    from cuda_ext import VARIANTS, VARIANTS_INT8

    fp16_fn = VARIANTS["warp"]
    int8_fn = VARIANTS_INT8[variant]

    torch.manual_seed(0)
    device = "cuda"
    num_seqs = 2
    lens = _make_lens(clen, num_seqs, device)
    max_len = int(lens.max())
    scale = 1.0 / math.sqrt(HEAD_DIM)

    q = torch.randn(num_seqs, num_heads, HEAD_DIM, dtype=torch.float16, device=device)
    k, v = _rand_kv(num_seqs, max_len, num_kv_heads, torch.float16, device)

    # FP16 baseline output.
    k_cache, v_cache, block_table = build_paged_kv_cache(k, v, lens, block_size=BLOCK_SIZE, x=X)
    out_fp16 = torch.empty_like(q)
    fp16_fn(out_fp16, q, k_cache, v_cache, block_table, lens, scale, BLOCK_SIZE)
    fp16 = out_fp16.float()

    def rel_err(mode):
        kc, vc, ks, vs, bt = build_paged_kv_cache_int8(
            k, v, lens, block_size=BLOCK_SIZE, x=X, mode=mode
        )
        out = torch.empty_like(q)
        int8_fn(out, q, kc, vc, ks, vs, bt, lens, scale, BLOCK_SIZE)
        diff = (out.float() - fp16).abs()
        denom = fp16.abs().clamp_min(1e-3)
        rel = diff / denom
        return rel.max().item(), rel.mean().item()

    pt_max, pt_mean = rel_err("per_token")
    ptn_max, ptn_mean = rel_err("per_tensor")
    print(f"\n[{variant} clen={clen} {num_heads}/{num_kv_heads}] "
          f"per_token max={pt_max:.4f} mean={pt_mean:.4f} | "
          f"per_tensor max={ptn_max:.4f} mean={ptn_mean:.4f}")

    # Loose sanity bound on per-token mean relative error.
    assert pt_mean < 0.2, f"per-token mean rel-err too high: {pt_mean}"
    # Per-token should be no worse than per-tensor on the mean.
    assert pt_mean <= ptn_mean + 1e-3, (
        f"per-token mean {pt_mean} should be <= per-tensor mean {ptn_mean}")


# --- Boundary cases & finite-output guards -----------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("variant", _variant_names() if torch.cuda.is_available() else [])
@pytest.mark.parametrize("clen", [1, 17, 4096])
def test_outputs_finite(variant, clen):
    """No NaN/Inf in the output across the variants, including the smallest and
    largest contexts. The cheap regression net for softmax-denominator bugs."""
    from cuda_ext import VARIANTS

    paged_decode_attention = VARIANTS[variant]
    torch.manual_seed(0)
    device = "cuda"
    num_heads, num_kv_heads, num_seqs = 8, 2, 2
    lens = _make_lens(clen, num_seqs, device)
    max_len = int(lens.max())
    scale = 1.0 / math.sqrt(HEAD_DIM)

    q = torch.randn(num_seqs, num_heads, HEAD_DIM, dtype=torch.float16, device=device)
    k, v = _rand_kv(num_seqs, max_len, num_kv_heads, torch.float16, device)
    k_cache, v_cache, block_table = build_paged_kv_cache(
        k, v, lens, block_size=BLOCK_SIZE, x=X)

    out = torch.empty_like(q)
    paged_decode_attention(out, q, k_cache, v_cache, block_table, lens, scale, BLOCK_SIZE)
    assert torch.isfinite(out).all(), f"{variant} produced non-finite output"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("variant", _variant_names() if torch.cuda.is_available() else [])
def test_empty_context_rejected(variant):
    """A context length of 0 has no tokens to attend to — the softmax denominator
    is 0, so the kernels must reject it rather than silently emit NaN. Decode never
    produces ctx=0 (a sequence always has >=1 token), so this is a precondition."""
    from cuda_ext import VARIANTS

    paged_decode_attention = VARIANTS[variant]
    torch.manual_seed(0)
    device = "cuda"
    num_heads, num_kv_heads = 8, 2
    # One real sequence plus one empty (ctx=0) sequence in the batch.
    lens = torch.tensor([8, 0], dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    q = torch.randn(2, num_heads, HEAD_DIM, dtype=torch.float16, device=device)
    k, v = _rand_kv(2, 8, num_kv_heads, torch.float16, device)
    k_cache, v_cache, block_table = build_paged_kv_cache(
        k, v, torch.tensor([8, 1], dtype=torch.int32, device=device),
        block_size=BLOCK_SIZE, x=X)

    out = torch.empty_like(q)
    with pytest.raises(RuntimeError):
        paged_decode_attention(out, q, k_cache, v_cache, block_table, lens, scale, BLOCK_SIZE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("variant", _int8_variant_names() if torch.cuda.is_available() else [])
@pytest.mark.parametrize("clen", [1, 17, 4096])
def test_int8_outputs_finite(variant, clen):
    """INT8 kernels must also be NaN/Inf-free across the context range."""
    from cuda_ext import VARIANTS_INT8

    int8_fn = VARIANTS_INT8[variant]
    torch.manual_seed(0)
    device = "cuda"
    num_heads, num_kv_heads, num_seqs = 8, 2, 2
    lens = _make_lens(clen, num_seqs, device)
    max_len = int(lens.max())
    scale = 1.0 / math.sqrt(HEAD_DIM)

    q = torch.randn(num_seqs, num_heads, HEAD_DIM, dtype=torch.float16, device=device)
    k, v = _rand_kv(num_seqs, max_len, num_kv_heads, torch.float16, device)
    kc, vc, ks, vs, bt = build_paged_kv_cache_int8(
        k, v, lens, block_size=BLOCK_SIZE, x=X, mode="per_token")

    out = torch.empty_like(q)
    int8_fn(out, q, kc, vc, ks, vs, bt, lens, scale, BLOCK_SIZE)
    assert torch.isfinite(out).all(), f"{variant} produced non-finite output"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("variant", _int8_variant_names() if torch.cuda.is_available() else [])
def test_int8_empty_context_rejected(variant):
    """INT8 kernels reject ctx=0 the same way the FP16 ones do."""
    from cuda_ext import VARIANTS_INT8

    int8_fn = VARIANTS_INT8[variant]
    torch.manual_seed(0)
    device = "cuda"
    num_heads, num_kv_heads = 8, 2
    lens = torch.tensor([8, 0], dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    q = torch.randn(2, num_heads, HEAD_DIM, dtype=torch.float16, device=device)
    k, v = _rand_kv(2, 8, num_kv_heads, torch.float16, device)
    kc, vc, ks, vs, bt = build_paged_kv_cache_int8(
        k, v, torch.tensor([8, 1], dtype=torch.int32, device=device),
        block_size=BLOCK_SIZE, x=X, mode="per_token")

    out = torch.empty_like(q)
    with pytest.raises(RuntimeError):
        int8_fn(out, q, kc, vc, ks, vs, bt, lens, scale, BLOCK_SIZE)


def _cosine(a, b):
    """Cosine similarity of two flattened tensors (stable near-zero metric)."""
    a, b = a.reshape(-1), b.reshape(-1)
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("variant", _int8_variant_names() if torch.cuda.is_available() else [])
def test_int8_precision_outlier_distribution(variant):
    """Stress the quant under heavy-tailed K (a few 'massive activation' channels,
    as in post-RoPE K) rather than benign N(0,1). Per-token quant should still beat
    per-tensor — and harder so, since per-tensor's single scale is wrecked by the
    outlier — and stay close to FP16 by cosine similarity (a stable metric that
    doesn't blow up on near-zero outputs the way per-element rel-err does)."""
    from cuda_ext import VARIANTS, VARIANTS_INT8

    fp16_fn = VARIANTS["warp"]
    int8_fn = VARIANTS_INT8[variant]

    torch.manual_seed(0)
    device = "cuda"
    num_heads, num_kv_heads, num_seqs, clen = 8, 2, 2, 500
    lens = _make_lens(clen, num_seqs, device)
    max_len = int(lens.max())
    scale = 1.0 / math.sqrt(HEAD_DIM)

    q = torch.randn(num_seqs, num_heads, HEAD_DIM, dtype=torch.float16, device=device)
    k, v = _rand_kv(num_seqs, max_len, num_kv_heads, torch.float16, device)
    # Inject outlier channels into K (a handful of head_dim positions ~20x larger).
    outlier_ch = torch.tensor([3, 50, 97], device=device)
    k[:, :, :, outlier_ch] *= 20.0

    k_cache, v_cache, bt = build_paged_kv_cache(k, v, lens, block_size=BLOCK_SIZE, x=X)
    out_fp16 = torch.empty_like(q)
    fp16_fn(out_fp16, q, k_cache, v_cache, bt, lens, scale, BLOCK_SIZE)
    fp16 = out_fp16.float()

    def cos(mode):
        kc, vc, ks, vs, bt8 = build_paged_kv_cache_int8(
            k, v, lens, block_size=BLOCK_SIZE, x=X, mode=mode)
        out = torch.empty_like(q)
        int8_fn(out, q, kc, vc, ks, vs, bt8, lens, scale, BLOCK_SIZE)
        return _cosine(out.float(), fp16)

    pt_cos, ptn_cos = cos("per_token"), cos("per_tensor")
    print(f"\n[{variant} outlier-K] per_token cos={pt_cos:.5f} "
          f"per_tensor cos={ptn_cos:.5f}")
    # Per-token tracks FP16 closely even under outliers.
    assert pt_cos > 0.99, f"per-token cosine too low under outliers: {pt_cos}"
    # And it is at least as faithful as per-tensor (the ablation point).
    assert pt_cos >= ptn_cos - 1e-4, (
        f"per-token cos {pt_cos} should be >= per-tensor cos {ptn_cos}")
