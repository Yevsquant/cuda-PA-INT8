"""Tests for the standalone Qwen INT8 decode bridge (plan §7).

Gate 1 (here, cheap, no model): the PagedKVCache write paths (prefill scatter +
decode append) produce a cache the custom kernels read correctly, at Qwen's head
config (12 Q / 2 KV heads, head_dim 128, GQA-6) — checked against the dequant
reference oracle.

Gate 2 (needs the model, slow): the FP16 bridge reproduces HF greedy decoding
token-for-token. Marked `slow`; run with `pytest -m slow`.
"""

import math
import os
import sys

import pytest
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "tests"))
sys.path.insert(0, os.path.join(_REPO, "benchmarks", "int8_bridge"))

from paged_decode_attn import (  # noqa: E402
    paged_decode_attention_reference,
    dequantize_kv,
)

HEAD_DIM = 128
BLOCK_SIZE = 16
X = 8
NUM_Q_HEADS = 12   # Qwen2.5-1.5B
NUM_KV_HEADS = 2   # GQA-6

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _kernel_out(cache, q, scale):
    """Run the right custom kernel for `cache`'s variant on layer 0."""
    from cuda_ext import VARIANTS, VARIANTS_INT8
    out = torch.empty_like(q)
    ctx = cache.context_lens(0)
    if cache.is_int8:
        VARIANTS_INT8["splitk_int8"](
            out, q, cache.k_cache[0], cache.v_cache[0],
            cache.k_scales[0], cache.v_scales[0],
            cache.block_table, ctx, scale, BLOCK_SIZE)
    else:
        VARIANTS["splitk"](
            out, q, cache.k_cache[0], cache.v_cache[0],
            cache.block_table, ctx, scale, BLOCK_SIZE)
    return out


def _reference_out(cache, q, scale):
    """Reference attention over the (dequantized) cache contents on layer 0."""
    if cache.is_int8:
        k_deq, v_deq = dequantize_kv(
            cache.k_cache[0], cache.v_cache[0], cache.k_scales[0], cache.v_scales[0])
    else:
        k_deq, v_deq = cache.k_cache[0].float(), cache.v_cache[0].float()
    return paged_decode_attention_reference(
        q.float(), k_deq, v_deq, cache.block_table, cache.context_lens(0),
        block_size=BLOCK_SIZE)


@needs_cuda
@pytest.mark.parametrize("variant", ["fp16", "int8_per_token", "int8_per_tensor"])
@pytest.mark.parametrize("clen", [1, 16, 17, 31, 128, 500])
def test_prefill_path_matches_reference(variant, clen):
    """Scatter `clen` prompt tokens via prefill(), then the kernel must match the
    dequant reference."""
    from paged_cache import PagedKVCache

    torch.manual_seed(0)
    dev = "cuda"
    scale = 1.0 / math.sqrt(HEAD_DIM)
    k = torch.randn(clen, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
    v = torch.randn(clen, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
    q = torch.randn(1, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)

    cache = PagedKVCache(
        num_layers=1, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, max_len=clen,
        block_size=BLOCK_SIZE, x=X, variant=variant, device=dev)
    cache.prefill(0, k, v)

    out = _kernel_out(cache, q, scale)
    ref = _reference_out(cache, q, scale)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@needs_cuda
def test_int8_cache_smaller_than_fp16():
    """Gate 4 (unit): the int8 cache (incl. fp32 scale buffers) is meaningfully
    smaller than the fp16 cache for the same geometry — expect ~0.5x + overhead."""
    from paged_cache import PagedKVCache
    kw = dict(num_layers=28, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
              max_len=2048, block_size=BLOCK_SIZE, x=X, device="cuda")
    fp16 = PagedKVCache(variant="fp16", **kw).kv_bytes()
    int8 = PagedKVCache(variant="int8_per_token", **kw).kv_bytes()
    assert int8 < fp16
    assert int8 / fp16 < 0.6, f"int8/fp16 ratio {int8/fp16:.3f} not near 0.5"


@needs_cuda
@pytest.mark.parametrize("variant", ["fp16", "int8_per_token", "int8_per_tensor"])
@pytest.mark.parametrize("clen", [17, 31, 128, 500])
def test_decode_append_path_matches_reference(variant, clen):
    """Prefill clen-1 tokens then append the last via decode_append(): exercises
    the incremental single-token write path the bridge uses every decode step."""
    from paged_cache import PagedKVCache

    torch.manual_seed(0)
    dev = "cuda"
    scale = 1.0 / math.sqrt(HEAD_DIM)
    k = torch.randn(clen, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
    v = torch.randn(clen, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
    q = torch.randn(1, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)

    cache = PagedKVCache(
        num_layers=1, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, max_len=clen,
        block_size=BLOCK_SIZE, x=X, variant=variant, device=dev)
    cache.prefill(0, k[:-1], v[:-1])
    new_len = cache.decode_append(0, k[-1], v[-1])
    assert new_len == clen

    out = _kernel_out(cache, q, scale)
    ref = _reference_out(cache, q, scale)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


# --- Gate 2: FP16 bridge reproduces HF greedy decoding (slow, needs model) ---

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
_PROMPTS = [
    "The capital of France is",
    "Q: What is 17 plus 26? A:",
]


@pytest.mark.slow
@needs_cuda
def test_fp16_bridge_matches_hf_greedy():
    """The non-quantized bridge (FP16 kernel + paged cache, prefill via SDPA) must
    reproduce HF's own greedy generation token-for-token. This proves the paged
    cache + decode loop are correct *before* any quantization is trusted."""
    import torch as _t
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import qwen_paged_attn as bridge
    from paged_cache import PagedKVCache

    tok = AutoTokenizer.from_pretrained(MODEL)
    bridge.register()
    # fp16 (not bf16): the kernel and paged cache are __half, so loading the
    # model in fp16 makes this an exact correctness check of the cache+loop+
    # kernel, isolated from the documented fp16-kernel-vs-bf16-model precision
    # caveat (run_bridge.py loads bf16; that gap is precision, not a bug).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=_t.float16, attn_implementation="sdpa").cuda().eval()
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    max_new = 32

    for p in _PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.cuda()

        model.set_attn_implementation("sdpa")
        # Qwen's generation_config defaults apply repetition_penalty=1.1 even
        # under do_sample=False, which rewrites logits away from pure greedy.
        # generate_greedy() does plain argmax, so compare against plain argmax.
        ref = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             repetition_penalty=1.0)
        ref_new = ref[0, ids.shape[1]:].cpu()

        model.set_attn_implementation(bridge.ATTN_NAME)
        cache = PagedKVCache(
            num_layers=cfg.num_hidden_layers, num_kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim, max_len=ids.shape[1] + max_new, variant="fp16")
        gen, _, _, _ = bridge.generate_greedy(model, ids, max_new, cache)
        gen = gen[0].cpu()

        n = min(len(gen), len(ref_new))
        match = (gen[:n] == ref_new[:n]).float().mean().item()
        print(f"\n[{p!r}] greedy match {match:.3f}\n  ref={ref_new[:n].tolist()}"
              f"\n  brg={gen[:n].tolist()}")
        assert match == 1.0, f"bridge diverged from HF greedy (match={match})"
