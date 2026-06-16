"""Route Qwen2 decode-step attention through the custom paged-decode kernels.

Registers a transformers 5.x attention interface that replaces the *core* attention
math (q_proj/RoPE/o_proj stay native, since the interface is called post-RoPE):

  - prefill (q_len > 1): full-precision causal SDPA for the output, and scatters the
    prompt K/V into the active PagedKVCache (quantized for the int8 variants).
  - decode  (q_len == 1): appends the new token's K/V to the cache, then calls the
    custom split-K kernel (FP16 or INT8) for the attention output.

batch=1, single sequence (plan §7). The active cache is a module global set per
generate call; generation is a manual greedy loop with use_cache=False and explicit
position_ids (so the model feeds correct RoPE positions and the interface only ever
sees the current chunk, never HF's own concatenated cache).
"""

import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers.modeling_utils import AttentionInterface

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "tests"))
from cuda_ext import VARIANTS, VARIANTS_INT8  # noqa: E402

ATTN_NAME = "paged_int8_bridge"

_ACTIVE = None  # the PagedKVCache the interface writes to / reads from


def set_active_cache(cache):
    global _ACTIVE
    _ACTIVE = cache


def paged_bridge_attention(module, query, key, value, attention_mask,
                           scaling=None, dropout=0.0, **kwargs):
    """Custom attention interface. Shapes (transformers convention):
        query: [B, num_q_heads,  q_len, head_dim]
        key/value: [B, num_kv_heads, q_len, head_dim]   (post-RoPE, pre-GQA-repeat)
    Returns (attn_output [B, q_len, num_q_heads, head_dim], attn_weights=None)."""
    cache = _ACTIVE
    assert cache is not None, "no active PagedKVCache; call set_active_cache()"
    layer = module.layer_idx
    B, _, q_len, _ = query.shape
    assert B == 1, "bridge is batch=1 only"

    if q_len > 1:
        # Prefill: scatter prompt K/V, output via full-precision causal SDPA.
        k = key[0].transpose(0, 1).contiguous()    # [q_len, num_kv_heads, head_dim]
        v = value[0].transpose(0, 1).contiguous()
        cache.prefill(layer, k, v)
        attn = F.scaled_dot_product_attention(
            query, key, value, is_causal=True, scale=scaling, enable_gqa=True)
        return attn.transpose(1, 2).contiguous(), None

    # Decode: append the single new token, then the custom kernel.
    cache.decode_append(layer, key[0, :, 0, :], value[0, :, 0, :])
    q = query[0].transpose(0, 1).contiguous().to(torch.float16)  # [1, num_q_heads, D]
    out = torch.empty_like(q)
    ctx = cache.context_lens(layer)
    if cache.is_int8:
        VARIANTS_INT8["splitk_int8"](
            out, q, cache.k_cache[layer], cache.v_cache[layer],
            cache.k_scales[layer], cache.v_scales[layer],
            cache.block_table, ctx, scaling, cache.block_size)
    else:
        VARIANTS["splitk"](
            out, q, cache.k_cache[layer], cache.v_cache[layer],
            cache.block_table, ctx, scaling, cache.block_size)
    return out.unsqueeze(1).to(query.dtype), None  # [1, 1, num_q_heads, D]


_REGISTERED = False


def register():
    """Register the interface once. Load the model with attn_implementation=ATTN_NAME
    (or set model.config._attn_implementation = ATTN_NAME) to activate it."""
    global _REGISTERED
    if not _REGISTERED:
        AttentionInterface.register(ATTN_NAME, paged_bridge_attention)
        _REGISTERED = True


@torch.no_grad()
def generate_greedy(model, input_ids, max_new_tokens, cache):
    """Greedy decode through the bridge cache. Returns
    (generated_ids [1, n], ttft_s, tpot_s, per_step_logits list).

    per_step_logits are the next-token logit vectors (prefill-last + each decode
    step) on CPU float32 — used downstream for the perplexity/precision metric.
    """
    device = input_ids.device
    cache.reset()
    set_active_cache(cache)
    prompt_len = input_ids.shape[1]

    pos = torch.arange(prompt_len, device=device).unsqueeze(0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model(input_ids=input_ids, position_ids=pos, use_cache=False).logits[:, -1, :]
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    step_logits = [logits[0].float().cpu()]
    next_tok = logits.argmax(-1, keepdim=True)
    generated = [next_tok]

    step_times = []
    cur = prompt_len
    for _ in range(max_new_tokens - 1):
        pos = torch.tensor([[cur]], device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(input_ids=next_tok, position_ids=pos, use_cache=False).logits[:, -1, :]
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        step_logits.append(logits[0].float().cpu())
        next_tok = logits.argmax(-1, keepdim=True)
        generated.append(next_tok)
        cur += 1

    tpot = sum(step_times) / len(step_times) if step_times else float("nan")
    return torch.cat(generated, dim=1), ttft, tpot, step_logits


@torch.no_grad()
def teacher_force_logprobs(model, prompt_ids, cont_ids, cache):
    """Per-token log-probabilities `cache`'s variant assigns to a fixed
    continuation `cont_ids` ([1, n]) after `prompt_ids`, via the same
    prefill+decode path as generate_greedy but with tokens forced. Returns a
    python list of n natural-log values. The cross-variant ppl/precision metric
    is computed from these by combine_bridge.py (each variant teacher-forces the
    fp16 continuation, so the runs stay independent / one-at-a-time in RAM)."""
    device = prompt_ids.device
    cache.reset()
    set_active_cache(cache)
    prompt_len = prompt_ids.shape[1]

    pos = torch.arange(prompt_len, device=device).unsqueeze(0)
    logits = model(input_ids=prompt_ids, position_ids=pos, use_cache=False).logits[:, -1, :]

    cont = cont_ids[0]
    logps = []
    cur = prompt_len
    for i in range(cont.shape[0]):
        logp = torch.log_softmax(logits[0].float(), dim=-1)
        logps.append(logp[cont[i]].item())
        tok = cont[i].view(1, 1)
        pos = torch.tensor([[cur]], device=device)
        logits = model(input_ids=tok, position_ids=pos, use_cache=False).logits[:, -1, :]
        cur += 1

    return logps


def teacher_force_ppl(model, prompt_ids, cont_ids, cache):
    """Perplexity `cache`'s variant assigns to `cont_ids` after `prompt_ids`."""
    logps = teacher_force_logprobs(model, prompt_ids, cont_ids, cache)
    return float(torch.exp(torch.tensor(-sum(logps) / len(logps))))
