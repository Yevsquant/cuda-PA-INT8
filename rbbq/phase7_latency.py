"""RBBQ pivot, Phase 7 — cost/latency of selective-granularity W8A8.

The pivot's premise is "C2-level accuracy at lower cost than all-dynamic." So far
cost was a proxy (# dynamic linears). This measures real wall-clock with int8 GEMMs
(torch._int_mm) on Qwen2.5-1.5B's actual linear shapes, isolating the only thing
selective changes: per-linear activation quant is static-per-tensor (precomputed
scalar scale) vs per-token-dynamic (runtime rowmax reduction + per-token scale).

Per linear we time: bf16 matmul; int8 static (static-quant + int8 GEMM + dequant);
int8 dynamic (dynamic-quant + int8 GEMM + dequant). Then aggregate a Qwen decoder
layer (7 linears) under all-static / selective (down_proj dynamic) / all-dynamic.

Note: torch._int_mm requires m>16, so this covers prefill / batched (compute-bound)
regimes; single-token decode int8 GEMM needs a different kernel (out of scope).
"""
import json
import torch

DEV = "cuda"
# Qwen2.5-1.5B: hidden 1536, intermediate 8960, 12 q heads (1536), 2 kv heads (256)
LAYER = [  # (name, K_in, N_out)
    ("q_proj", 1536, 1536), ("k_proj", 1536, 256), ("v_proj", 1536, 256),
    ("o_proj", 1536, 1536), ("gate_proj", 1536, 8960), ("up_proj", 1536, 8960),
    ("down_proj", 8960, 1536),
]


def timed(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # microseconds


def bench(T, K, N):
    x = torch.randn(T, K, device=DEV, dtype=torch.bfloat16)
    wq = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=DEV)  # [N,K] weight
    wbf = torch.randn(N, K, device=DEV, dtype=torch.bfloat16)
    inv_s = torch.tensor(8.0, device=DEV)            # precomputed static scale (1/scale)
    sw = torch.rand(N, device=DEV) + 0.1             # per-channel weight scale

    def q_static():
        return torch.clamp(torch.round(x * inv_s), -127, 127).to(torch.int8)

    def q_dynamic():
        s = (x.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-8)
        return torch.clamp(torch.round(x / s), -127, 127).to(torch.int8), s

    def gemm(a):
        return torch._int_mm(a, wq.t())

    def f_bf16():
        return x @ wbf.t()

    def f_static():
        a = q_static()
        c = gemm(a)
        return c.float() * sw  # dequant (static act scale folded into sw here)

    def f_dynamic():
        a, s = q_dynamic()
        c = gemm(a)
        return c.float() * sw * s  # per-token act scale

    return dict(
        bf16=timed(f_bf16), static=timed(f_static), dynamic=timed(f_dynamic),
        gemm=timed(lambda: gemm(q_static())),
        # the ONLY thing that differs static->dynamic: the per-token rowmax reduction
        reduce=timed(lambda: x.abs().amax(-1, keepdim=True)),
    )


def main():
    out = {}
    for T in (2048, 128):
        per = {name: bench(T, K, N) for name, K, N in LAYER}
        # per-layer aggregation under each policy
        def layer_sum(mode_of):
            return sum(per[n][mode_of(n)] for n, _, _ in LAYER)
        agg = dict(
            all_bf16=sum(per[n]["bf16"] for n, _, _ in LAYER),
            all_static=layer_sum(lambda n: "static"),
            selective=layer_sum(lambda n: "dynamic" if n == "down_proj" else "static"),
            all_dynamic=layer_sum(lambda n: "dynamic"),
        )
        # honest marginal cost of per-token dynamic = the rowmax reduction only
        gemm_total = sum(per[n]["gemm"] for n, _, _ in LAYER)
        red_sel = per["down_proj"]["reduce"]
        red_all = sum(per[n]["reduce"] for n, _, _ in LAYER)
        agg["reduce_selective"] = red_sel
        agg["reduce_all_dynamic"] = red_all
        agg["gemm_total"] = gemm_total
        out[f"T{T}"] = dict(per_linear=per, layer_us=agg)
        print(f"=== T={T} (per-layer, 7 linears, microseconds) ===")
        print(f"  int8 GEMM total (all linears)        {gemm_total:8.1f} us")
        print(f"  marginal dynamic cost (rowmax reduce): selective(down_proj)={red_sel:.1f}"
              f"  all_dynamic(7 linears)={red_all:.1f}")
        print(f"  -> selective adds {red_sel/gemm_total*100:.1f}% over all-static GEMM; "
              f"all-dynamic adds {red_all/gemm_total*100:.1f}%")
        print(f"  [unfused-eager pipeline, NOT representative: all_bf16={agg['all_bf16']:.0f} "
              f"all_static={agg['all_static']:.0f} all_dynamic={agg['all_dynamic']:.0f}]")
    json.dump(out, open("report/phase7_latency.json", "w"), indent=2)
    print("wrote report/phase7_latency.json")


if __name__ == "__main__":
    main()
