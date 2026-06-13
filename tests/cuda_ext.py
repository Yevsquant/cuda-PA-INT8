"""JIT-compile and expose the naive paged-attention CUDA kernel.

The .cu source lives in kernels/; this loader compiles it on first import via
ninja (torch.utils.cpp_extension.load) and re-exports `paged_decode_attention`.
"""

import os
import functools

import torch
from torch.utils.cpp_extension import load

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_KERNELS = os.path.join(_REPO_ROOT, "kernels")
_NAIVE_SRC = os.path.join(_KERNELS, "paged_attention_naive.cu")

# Optimized stages: compiled into one extension alongside bindings.cpp so every
# variant is callable in a single process. Each entry maps a variant name to its
# op name on the extension module.
_OPT_SOURCES = [
    "bindings.cpp",
    "paged_attention_v1_vectorized.cu",
    "paged_attention_v2_online.cu",
    "paged_attention_v3_warp.cu",
    "paged_attention_v4_splitk.cu",
]
_OPT_OPS = {
    "vec": "paged_decode_attn_vec",
    "online": "paged_decode_attn_online",
    "warp": "paged_decode_attn_warp",
    "splitk": "paged_decode_attn_splitk",
}


@functools.lru_cache(maxsize=1)
def _naive_ext():
    return load(
        name="paged_attn_naive",
        sources=[_NAIVE_SRC],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )


@functools.lru_cache(maxsize=1)
def _opt_ext():
    return load(
        name="paged_attn_opt",
        sources=[os.path.join(_KERNELS, s) for s in _OPT_SOURCES],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )


def paged_decode_attention(out, q, k_cache, v_cache, block_table, context_lens,
                           scale, block_size=16):
    _naive_ext().paged_decode_attention(
        out, q, k_cache, v_cache, block_table, context_lens, scale, block_size
    )
    return out


def _make_variant(op_name):
    def run(out, q, k_cache, v_cache, block_table, context_lens, scale,
            block_size=16):
        getattr(_opt_ext(), op_name)(
            out, q, k_cache, v_cache, block_table, context_lens, scale, block_size
        )
        return out
    run.__name__ = op_name
    return run


# Registry of all callable variants (name -> fn with the naive signature).
VARIANTS = {"naive": paged_decode_attention}
VARIANTS.update({name: _make_variant(op) for name, op in _OPT_OPS.items()})
