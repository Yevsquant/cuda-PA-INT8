"""JIT-compile and expose the naive paged-attention CUDA kernel.

The .cu source lives in kernels/; this loader compiles it on first import via
ninja (torch.utils.cpp_extension.load) and re-exports `paged_decode_attention`.
"""

import os
import functools

import torch
from torch.utils.cpp_extension import load

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "kernels", "paged_attention_naive.cu")


@functools.lru_cache(maxsize=1)
def _ext():
    return load(
        name="paged_attn_naive",
        sources=[_SRC],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )


def paged_decode_attention(out, q, k_cache, v_cache, block_table, context_lens,
                           scale, block_size=16):
    _ext().paged_decode_attention(
        out, q, k_cache, v_cache, block_table, context_lens, scale, block_size
    )
    return out
