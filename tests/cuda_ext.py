"""JIT-compile and expose the naive paged-attention CUDA kernel.

The .cu source lives in kernels/; this loader compiles it on first import via
ninja (torch.utils.cpp_extension.load) and re-exports `paged_decode_attention`.
"""

import os
import glob
import functools

# Cap ninja's compile parallelism BEFORE torch is imported. By default
# cpp_extension launches one nvcc per CPU (24 here); each -O3 compile of the
# heavy split-K / INT8 kernels needs a few GB, so the parallel spike blows past
# this JupyterHub pod's 8 GiB cgroup cap and the pod gets OOM-killed mid-build
# (looks like a "disconnect"). Serializing keeps peak memory well under the cap.
# Overridable: export MAX_JOBS=N before running.
os.environ.setdefault("MAX_JOBS", "2")

import torch
from torch.utils.cpp_extension import load, _get_build_directory

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
    "paged_attention_v5_warp_int8.cu",
    "paged_attention_v6_splitk_int8.cu",
]
_OPT_OPS = {
    "vec": "paged_decode_attn_vec",
    "online": "paged_decode_attn_online",
    "warp": "paged_decode_attn_warp",
    "splitk": "paged_decode_attn_splitk",
}
# INT8 KV-cache variants. Separate registry because the call signature carries
# the extra k_scales / v_scales buffers.
_OPT_OPS_INT8 = {
    "warp_int8": "paged_decode_attn_warp_int8",
    "splitk_int8": "paged_decode_attn_splitk_int8",
}


def _compiler_running():
    """True if any ninja/nvcc compiler process is alive on this machine."""
    names = {"ninja", "nvcc", "cicc", "ptxas", "cudafe++"}
    for comm in glob.glob("/proc/[0-9]*/comm"):
        try:
            with open(comm) as f:
                if f.read().strip() in names:
                    return True
        except OSError:
            continue
    return False


def _clear_stale_lock(name):
    """Remove a leftover cpp_extension baton lock from a killed/crashed build.

    `load` serializes builds with a `lock` file in the build dir: a process
    holds it while ninja runs. If that process is killed before releasing,
    the lock lingers and the next `load` blocks forever polling for it. Treat
    the lock as stale only when no compiler is actually running, so we never
    yank the baton from a live build. Best-effort — never let this block load.
    """
    try:
        lock = os.path.join(_get_build_directory(name, verbose=False), "lock")
        if os.path.exists(lock) and not _compiler_running():
            os.remove(lock)
    except OSError:
        pass


@functools.lru_cache(maxsize=1)
def _naive_ext():
    _clear_stale_lock("paged_attn_naive")
    return load(
        name="paged_attn_naive",
        sources=[_NAIVE_SRC],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )


@functools.lru_cache(maxsize=1)
def _opt_ext():
    _clear_stale_lock("paged_attn_opt")
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


def _make_variant_int8(op_name):
    def run(out, q, k_cache, v_cache, k_scales, v_scales, block_table,
            context_lens, scale, block_size=16):
        getattr(_opt_ext(), op_name)(
            out, q, k_cache, v_cache, k_scales, v_scales, block_table,
            context_lens, scale, block_size
        )
        return out
    run.__name__ = op_name
    return run


# Registry of INT8 KV-cache variants (name -> fn carrying k_scales/v_scales).
VARIANTS_INT8 = {name: _make_variant_int8(op) for name, op in _OPT_OPS_INT8.items()}
