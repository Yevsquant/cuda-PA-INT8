"""RBBQ revised, Phase 15 — token-outlier fallback at down_proj.

Solution #4, a C1<->all-dynamic middle ground: static int8 activations for normal tokens
at down_proj, but route OUTLIER tokens (per-token rowmax > k * median_rowmax) through FP16
(int8 weight kept). The static scale is set to thr/127 so non-escalated tokens fit exactly.
Sweep k -> trade escalated-token fraction vs ppl. Tests whether the W8A8 damage is carried
by a small fraction of tokens. Qwen2.5-1.5B, WikiText-2.
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import phase4_qwen_decomp as p4
import phase5_selective as p5
from phase3_variant_b import FlexQuantLinear, q_pt
from transformers import AutoTokenizer
from datasets import load_dataset

DEV = "cuda"
C1 = dict(qw=True, qa=True, w_gran="tensor")


def is_down(n):
    return n.endswith("down_proj")


@torch.no_grad()
def median_rowmax(model, cal):
    samp = {}
    tg = [(n, m) for n, m in model.named_modules() if is_down(n)]

    def mk(n):
        def hook(mod, inp, out, n=n):
            rm = inp[0].detach().abs().amax(-1).reshape(-1).float()
            if rm.numel() > 8000:
                rm = rm[torch.randint(0, rm.numel(), (8000,), device=rm.device)]
            samp[n] = rm.cpu() if n not in samp else torch.cat([samp[n], rm.cpu()])[:200000]
        return hook
    h = [m.register_forward_hook(mk(n)) for n, m in tg]
    for i in range(8):
        model(cal[:, i * 2048:(i + 1) * 2048])
    for x in h:
        x.remove()
    return {n: float(v.median()) for n, v in samp.items()}


class TokenHybrid(nn.Module):
    """Static int8 act for normal tokens; FP16 act for tokens with rowmax > thr."""
    def __init__(self, lin, med, k):
        super().__init__()
        self.bias = lin.bias
        w = lin.weight.detach()
        sw = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127  # per-channel weight
        self.register_buffer("wq", q_pt(w, sw))
        self.thr = med * k
        self.frac_sum = 0.0
        self.frac_n = 0

    def forward(self, x):
        rowmax = x.detach().abs().amax(-1, keepdim=True)
        s = self.thr / 127.0
        xq = torch.clamp(torch.round(x / s), -127, 127) * s
        esc = rowmax > self.thr
        self.frac_sum += esc.float().mean().item()
        self.frac_n += 1
        act = torch.where(esc, x, xq)
        return F.linear(act, self.wq, self.bias)


def wrap_hybrid(model, med, k):
    qlins = []
    for name, mod in list(p4.targets(model)):
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        child = name.rsplit(".", 1)[1]
        if is_down(name):
            setattr(parent, child, TokenHybrid(mod, med[name], k))
        else:
            ql = FlexQuantLinear(mod, **C1)
            setattr(parent, child, ql)
            qlins.append(model.get_submodule(name))
    return qlins


def main():
    tok = AutoTokenizer.from_pretrained(p4.MODEL)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    cal = tok("\n\n".join(t for t in ds["train"]["text"] if t.strip()),
              return_tensors="pt").input_ids.to(DEV)
    ev = tok("\n\n".join(t for t in ds["test"]["text"] if t.strip()),
             return_tensors="pt").input_ids.to(DEV)
    fp = round(p4.ppl(p4.fresh(), ev), 3)
    m0 = p4.fresh()
    med = median_rowmax(m0, cal)
    del m0; torch.cuda.empty_cache()

    res = {"fp": fp}
    print(f"=== Phase 15 token-outlier fallback, Qwen (FP {fp}) ===")
    for k in [16, 8, 4, 2]:
        m = p4.fresh()
        ql = wrap_hybrid(m, med, k)
        p4.calibrate(m, ql, cal)
        p = round(p4.ppl(m, ev), 3)
        fr = sum(mod.frac_sum / mod.frac_n for _, mod in m.named_modules()
                 if isinstance(mod, TokenHybrid)) / sum(1 for _, mm in m.named_modules() if isinstance(mm, TokenHybrid))
        res[f"k{k}"] = {"ppl": p, "esc_frac": round(fr, 4)}
        print(f"  k={k:<3} esc_frac={fr:.4f}  ppl={p:.3f} (dppl={p-fp:+.3f})")
        del m; torch.cuda.empty_cache()
    json.dump(res, open("report/phase15_token_fallback.json", "w"), indent=2)
    print("wrote report/phase15_token_fallback.json")


if __name__ == "__main__":
    main()
