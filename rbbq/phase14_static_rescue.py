"""RBBQ revised, Phase 14 — residual-aware static-scale search ("static rescue").

Solution #3: max-scale static per-tensor fails at down_proj because one outlier token sets
the range. Try a constrained STATIC alternative (one scale per linear, still C1-deployable):
choose the down_proj activation clip by percentile or MSE-optimal clipping instead of max,
and see how much of the C1 gap closes WITHOUT per-token dynamic. Qwen2.5-1.5B, WikiText-2.

Candidates for the down_proj static scale (others stay max-calibrated C1):
  max (=plain) | p99.9 | p99.5 | p99 | MSE-optimal | (dynamic = upper bound)
"""
import json
import torch

import phase4_qwen_decomp as p4
import phase5_selective as p5
from phase3_variant_b import q_pt
from transformers import AutoTokenizer
from datasets import load_dataset

DEV = "cuda"
C1 = dict(qw=True, qa=True, w_gran="tensor")
DYN = p5.DYN
CAP = 120_000


def is_down(n):
    return n.endswith("down_proj")


@torch.no_grad()
def collect_down_samples(model, cal):
    """Capped sample of |x| at each down_proj input."""
    samp = {}
    tgts = [(n, m) for n, m in model.named_modules() if is_down(n)]

    def mk(n):
        def hook(mod, inp, out, n=n):
            v = inp[0].detach().abs().reshape(-1).float()
            if v.numel() > 20000:
                v = v[torch.randint(0, v.numel(), (20000,), device=v.device)]
            cur = samp.get(n)
            samp[n] = v.cpu() if cur is None else torch.cat([cur, v.cpu()])[:CAP]
        return hook
    h = [m.register_forward_hook(mk(n)) for n, m in tgts]
    for i in range(8):
        model(cal[:, i * 2048:(i + 1) * 2048])
    for x in h:
        x.remove()
    return samp


def mse_opt_scale(s):
    """1D search for clip c minimizing int8 quant MSE over sample s (=|x|)."""
    mx = s.max()
    best_c, best_e = mx, 1e30
    for f in torch.linspace(0.3, 1.0, 15):
        c = (mx * f).clamp_min(1e-8)
        sc = c / 127
        q = torch.clamp(torch.round(s / sc), 0, 127) * sc  # |x| domain
        e = ((q - s) ** 2).mean().item()
        if e < best_e:
            best_e, best_c = e, c
    return best_c


def scales_for(samp, kind):
    out = {}
    for n, s in samp.items():
        if kind == "max":
            out[n] = s.max()
        elif kind == "mse":
            out[n] = mse_opt_scale(s)
        else:  # percentile like p99.9
            out[n] = torch.quantile(s, float(kind))
    return out


def main():
    tok = AutoTokenizer.from_pretrained(p4.MODEL)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    cal = tok("\n\n".join(t for t in ds["train"]["text"] if t.strip()),
              return_tensors="pt").input_ids.to(DEV)
    ev = tok("\n\n".join(t for t in ds["test"]["text"] if t.strip()),
             return_tensors="pt").input_ids.to(DEV)
    fp = round(p4.ppl(p4.fresh(), ev), 3)

    m0 = p4.fresh()
    samp = collect_down_samples(m0, cal)
    mse_sc = scales_for(samp, "mse")          # per-linear MSE-optimal clip (abs value)
    del m0; torch.cuda.empty_cache()

    # One calibrated model; sweep down_proj static scale = fraction of TRUE max (reused)
    m = p4.fresh()
    ql = p4.wrap(m, lambda n: dict(C1))
    p4.calibrate(m, ql, cal)
    downs = [(n, mod) for n, mod in m.named_modules() if is_down(n)]
    true_max = {n: mod.act_scale.clone() for n, mod in downs}

    def eval_with(scale_of):
        for n, mod in downs:
            mod.act_scale = scale_of(n).to(DEV).float()
        return round(p4.ppl(m, ev), 3)

    res = {"fp": fp}
    for f in [1.0, 0.99, 0.97, 0.95, 0.90, 0.80, 0.70]:
        res[f"f{f}"] = eval_with(lambda n, f=f: true_max[n] * f)
        print(f"  down static = {f:.2f}*max  ppl={res[f'f{f}']:.3f} (dppl={res[f'f{f}']-fp:+.3f})")
    res["mse_opt"] = eval_with(lambda n: mse_sc[n])
    print(f"  down static = MSE-opt   ppl={res['mse_opt']:.3f} (dppl={res['mse_opt']-fp:+.3f})")
    del m; torch.cuda.empty_cache()
    # dynamic upper bound
    md = p4.fresh()
    qld = p4.wrap(md, lambda n: dict(DYN) if is_down(n) else dict(C1))
    p4.calibrate(md, qld, cal)
    res["dynamic"] = round(p4.ppl(md, ev), 3)
    print(f"  down dynamic            ppl={res['dynamic']:.3f} (dppl={res['dynamic']-fp:+.3f})")
    del md; torch.cuda.empty_cache()
    json.dump(res, open("report/phase14_static_rescue.json", "w"), indent=2)
    print("wrote report/phase14_static_rescue.json")


if __name__ == "__main__":
    main()
