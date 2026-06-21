"""RBBQ pivot, Phase 10 — is int8 GEMM fundamentally slow on Hopper, or just _int_mm?

Phase 9: torch._int_mm is 6-9x slower than bf16 on the H200. This phase tests whether
a tuned int8 kernel changes that, and what the real low-precision latency path on
Hopper is, by benchmarking on the Qwen linear shapes:
  bf16 (cuBLAS)  |  int8 torch._int_mm  |  int8 tuned Triton  |  fp8 torch._scaled_mm
Plus the hardware-agnostic memory-footprint win (the actual W8A8 benefit, esp. decode).

NOTE: project target is A100 (int8 IMMA well-tuned); only H200 measurable here.
"""
import json
import torch
import triton
import triton.language as tl
import phase7_latency as p7

DEV = "cuda"


@triton.autotune(
    configs=[
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'GM': 8}, num_warps=8, num_stages=3),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 64, 'GM': 8}, num_warps=8, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GM': 8}, num_warps=4, num_stages=4),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 128, 'GM': 8}, num_warps=8, num_stages=3),
        triton.Config({'BM': 64, 'BN': 256, 'BK': 64, 'GM': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _i8gemm(a_ptr, b_ptr, c_ptr, M, N, K,
            sam, sak, sbk, sbn, scm, scn,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    ng = GM * nn
    gid = pid // ng
    fr = gid * GM
    gsz = min(nm - fr, GM)
    pm = fr + ((pid % ng) % gsz)
    pn = (pid % ng) // gsz
    ram = (pm * BM + tl.arange(0, BM)) % M
    rbn = (pn * BN + tl.arange(0, BN)) % N
    rk = tl.arange(0, BK)
    a_ptr += ram[:, None] * sam + rk[None, :] * sak
    b_ptr += rk[:, None] * sbk + rbn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptr, mask=rk[None, :] < K - k * BK, other=0)
        b = tl.load(b_ptr, mask=rk[:, None] < K - k * BK, other=0)
        acc += tl.dot(a, b, out_dtype=tl.int32)
        a_ptr += BK * sak
        b_ptr += BK * sbk
    cm = pm * BM + tl.arange(0, BM)
    cn = pn * BN + tl.arange(0, BN)
    tl.store(c_ptr + cm[:, None] * scm + cn[None, :] * scn, acc,
             mask=(cm[:, None] < M) & (cn[None, :] < N))


def triton_i8(a, b):  # a[M,K] int8, b[K,N] int8 -> [M,N] int32
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, dtype=torch.int32, device=DEV)
    grid = lambda M_: (triton.cdiv(M, M_['BM']) * triton.cdiv(N, M_['BN']),)
    _i8gemm[grid](a, b, c, M, N, K, a.stride(0), a.stride(1),
                  b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c


def bench(T, K, N):
    abf = torch.randn(T, K, device=DEV, dtype=torch.bfloat16)
    bbf = torch.randn(K, N, device=DEV, dtype=torch.bfloat16)
    a8 = torch.randint(-127, 127, (T, K), dtype=torch.int8, device=DEV)
    b8 = torch.randint(-127, 127, (K, N), dtype=torch.int8, device=DEV)
    af8 = abf.to(torch.float8_e4m3fn)
    bf8 = bbf.to(torch.float8_e4m3fn).t().contiguous().t()  # [K,N] with col-major for scaled_mm
    s = torch.tensor(1.0, device=DEV)
    triton_i8(a8, b8)  # warm/autotune
    return dict(
        bf16=p7.timed(lambda: abf @ bbf),
        int_mm=p7.timed(lambda: torch._int_mm(a8, b8)),
        triton_i8=p7.timed(lambda: triton_i8(a8, b8)),
        fp8=p7.timed(lambda: torch._scaled_mm(af8, bf8, scale_a=s, scale_b=s, out_dtype=torch.bfloat16)),
    )


def main():
    out = {}
    for T in (2048, 128):
        out[f"T{T}"] = {n: bench(T, K, N) for n, K, N in p7.LAYER}
        print(f"=== T={T} GEMM us (x = vs bf16) ===")
        tot = {k: 0.0 for k in ["bf16", "int_mm", "triton_i8", "fp8"]}
        for n, _, _ in p7.LAYER:
            r = out[f"T{T}"][n]
            for k in tot:
                tot[k] += r[k]
            print(f"  {n:10s} bf16={r['bf16']:6.1f}  int_mm={r['int_mm']:6.1f}"
                  f"({r['int_mm']/r['bf16']:.1f}x)  triton_i8={r['triton_i8']:6.1f}"
                  f"({r['triton_i8']/r['bf16']:.1f}x)  fp8={r['fp8']:6.1f}({r['fp8']/r['bf16']:.1f}x)")
        print(f"  LAYER TOTAL: bf16={tot['bf16']:.0f}  int_mm={tot['int_mm']:.0f}"
              f"({tot['int_mm']/tot['bf16']:.1f}x)  triton_i8={tot['triton_i8']:.0f}"
              f"({tot['triton_i8']/tot['bf16']:.1f}x)  fp8={tot['fp8']:.0f}({tot['fp8']/tot['bf16']:.1f}x)")
    # memory footprint (weights+activations bytes), per decoder layer
    bytes_layer = {dt: sum((N * K + T2048 * K) * b for _, K, N in p7.LAYER)
                   for dt, b, T2048 in [("bf16", 2, 2048), ("int8/fp8", 1, 2048)]}
    out["mem_bytes_T2048"] = bytes_layer
    print("memory/layer (W+A bytes, T=2048):", {k: f"{v/1e6:.1f}MB" for k, v in bytes_layer.items()})
    json.dump(out, open("report/phase10_gemm.json", "w"), indent=2)
    print("wrote report/phase10_gemm.json")


if __name__ == "__main__":
    main()
