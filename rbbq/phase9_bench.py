"""Phase 9 benchmark — fused W8A8 (Triton) vs eager vs bf16, Qwen linear shapes.

Redoes Phase 7's measurement with the FUSED quant/dequant kernels, so the int8 path
reflects a real fused implementation instead of unfused eager elementwise overhead.
"""
import json
import torch
import phase7_latency as p7   # for timed() and LAYER shapes
import phase9_fused_w8a8 as k

DEV = "cuda"


def bench(T, K, N):
    x = torch.randn(T, K, device=DEV, dtype=torch.bfloat16)
    wq = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=DEV)
    wqt = wq.t().contiguous()
    wbf = torch.randn(N, K, device=DEV, dtype=torch.bfloat16)
    wsc = torch.rand(N, device=DEV) * 0.02 + 0.005
    inv = torch.tensor(8.0, device=DEV)

    # eager unfused int8 (Phase-7 style)
    def eager(dyn):
        if dyn:
            s = (x.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-8)
            a = torch.clamp(torch.round(x / s), -127, 127).to(torch.int8)
            return (torch._int_mm(a, wqt).float() * wsc * s)
        a = torch.clamp(torch.round(x * inv), -127, 127).to(torch.int8)
        return (torch._int_mm(a, wqt).float() * wsc)

    return dict(
        bf16=p7.timed(lambda: x @ wbf.t()),
        eager_static=p7.timed(lambda: eager(False)),
        eager_dynamic=p7.timed(lambda: eager(True)),
        fused_static=p7.timed(lambda: k.linear_fused(x, wqt, wsc, dynamic=False, inv_scale=8.0)),
        fused_dynamic=p7.timed(lambda: k.linear_fused(x, wqt, wsc, dynamic=True)),
    )


def main():
    out = {}
    for T in (2048, 128):
        per = {n: bench(T, K, N) for n, K, N in p7.LAYER}

        def layer(mode_of):
            return sum(per[n][mode_of(n)] for n, _, _ in p7.LAYER)
        agg = dict(
            bf16=layer(lambda n: "bf16"),
            eager_all_dynamic=layer(lambda n: "eager_dynamic"),
            fused_all_static=layer(lambda n: "fused_static"),
            fused_selective=layer(lambda n: "fused_dynamic" if n == "down_proj" else "fused_static"),
            fused_all_dynamic=layer(lambda n: "fused_dynamic"),
        )
        out[f"T{T}"] = dict(per_linear=per, layer_us=agg)
        print(f"=== T={T} (per-decoder-layer, 7 linears, microseconds) ===")
        for kk, vv in agg.items():
            rel = vv / agg["bf16"]
            print(f"  {kk:20s} {vv:8.1f} us   ({rel:.2f}x bf16)")
        dd = per["down_proj"]
        print(f"  down_proj: bf16={dd['bf16']:.1f} | eager_dyn={dd['eager_dynamic']:.1f} "
              f"-> fused_dyn={dd['fused_dynamic']:.1f} | fused_static={dd['fused_static']:.1f}")
    json.dump(out, open("report/phase9_bench.json", "w"), indent=2)
    print("wrote report/phase9_bench.json")


if __name__ == "__main__":
    main()
