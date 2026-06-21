"""RBBQ pivot, Phase 9 — Fused W8A8 kernel (Triton) for an honest latency number.

Phase 7's int8 path looked 4x slower than bf16 because eager quant/dequant ran as
separate kernels on large tensors. This fuses them with Triton:
  - prologue: per-token DYNAMIC quant (rowmax reduce + quantize) in ONE kernel,
    or STATIC quant (scalar scale, no reduce);
  - GEMM: torch._int_mm (cutlass int8);
  - epilogue: scaled dequant (int32 * act_scale[m] * weight_scale[n] -> bf16) in ONE
    kernel.
Then re-benchmark bf16 vs eager-int8 vs fused-int8, and static vs dynamic vs selective.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

DEV = "cuda"


@triton.jit
def _quant_dyn(x_ptr, q_ptr, s_ptr, K, stride_m, BLOCK_K: tl.constexpr):
    m = tl.program_id(0)
    amax = 0.0
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + m * stride_m + offs, mask=offs < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x)))
    scale = tl.where(amax > 0, amax / 127.0, 1e-8)
    tl.store(s_ptr + m, scale)
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + m * stride_m + offs, mask=offs < K, other=0.0).to(tl.float32)
        q = libdevice.rint(x / scale)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(q_ptr + m * K + offs, q.to(tl.int8), mask=offs < K)


@triton.jit
def _quant_static(x_ptr, q_ptr, inv_scale, K, stride_m, BLOCK_K: tl.constexpr):
    m = tl.program_id(0)
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + m * stride_m + offs, mask=offs < K, other=0.0).to(tl.float32)
        q = libdevice.rint(x * inv_scale)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(q_ptr + m * K + offs, q.to(tl.int8), mask=offs < K)


@triton.jit
def _dequant(c_ptr, as_ptr, ws_ptr, o_ptr, N, stride_m, has_as: tl.constexpr,
             BLOCK_N: tl.constexpr):
    m = tl.program_id(0)
    a = tl.load(as_ptr + m) if has_as else 1.0
    for n0 in range(0, N, BLOCK_N):
        offs = n0 + tl.arange(0, BLOCK_N)
        c = tl.load(c_ptr + m * stride_m + offs, mask=offs < N, other=0).to(tl.float32)
        w = tl.load(ws_ptr + offs, mask=offs < N, other=0.0)
        tl.store(o_ptr + m * N + offs, (c * a * w).to(tl.bfloat16), mask=offs < N)


def quant_dynamic(x):
    M, K = x.shape
    q = torch.empty(M, K, dtype=torch.int8, device=DEV)
    s = torch.empty(M, dtype=torch.float32, device=DEV)
    _quant_dyn[(M,)](x, q, s, K, x.stride(0), BLOCK_K=1024)
    return q, s


def quant_static(x, inv_scale):
    M, K = x.shape
    q = torch.empty(M, K, dtype=torch.int8, device=DEV)
    _quant_static[(M,)](x, q, float(inv_scale), K, x.stride(0), BLOCK_K=1024)
    return q


def dequant(c_i32, w_scale, a_scale=None):
    M, N = c_i32.shape
    o = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
    has = a_scale is not None
    _dequant[(M,)](c_i32, a_scale if has else c_i32, w_scale, o, N,
                   c_i32.stride(0), has_as=has, BLOCK_N=1024)
    return o


def linear_fused(x, wq_t, w_scale, dynamic=True, inv_scale=None, static_act=1.0):
    """x[M,K] bf16, wq_t[K,N] int8, w_scale[N]. Returns bf16 [M,N]."""
    if dynamic:
        q, s = quant_dynamic(x)
        return dequant(torch._int_mm(q, wq_t), w_scale, a_scale=s)
    q = quant_static(x, inv_scale)
    return dequant(torch._int_mm(q, wq_t), w_scale * static_act)
