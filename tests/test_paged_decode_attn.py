import math

import pytest
import torch

from paged_decode_attn import paged_decode_attention_reference, build_paged_kv_cache

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("clen", CONTEXT_LENS)
@pytest.mark.parametrize("num_heads,num_kv_heads", GQA)
@pytest.mark.parametrize("num_seqs", NUM_SEQS)
def test_cuda_matches_reference(clen, num_heads, num_kv_heads, num_seqs):
    """The naive CUDA kernel must match the reference within fp16 tolerance."""
    from cuda_ext import paged_decode_attention  # JIT-compiles on first call

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
