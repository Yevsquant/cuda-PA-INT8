"""RBBQ revised, Phase 12 — RBBQ-C consolidation + residual-diagnostic -> D linkage.

Closes solution #1: shows the original proposal's residual-add instrumentation and the
selective-granularity selector AGREE on the causal linear, then locks the group-aware
policy. Two per-linear signals on Qwen2.5-1.5B:
  - residual diagnostic: OUTPUT channel dominance = max_c E_out[c]/median_c E_out[c]
    (how much a linear concentrates energy into the residual it writes) -- the
    update-branch outputs (o_proj, down_proj) are what the residual add sees;
  - selection signal: INPUT per-token spread D (Phase 5).
Claim: both rank down_proj (the MLP update-branch output) highest -> the residual
diagnosis and D point to the same linear, justifying the RBBQ framing.

RBBQ-C policy (locked): per-token dynamic iff group == mlp_out OR D >= thr; else static.
"""
import json
import numpy as np
import torch

import phase4_qwen_decomp as p4
import phase5_selective as p5
from transformers import AutoTokenizer
from datasets import load_dataset

DEV = "cuda"
C1 = dict(qw=True, qa=True, w_gran="tensor")
DYN = p5.DYN


def collect_outdom(named_targets, run_calib):
    """Per-linear OUTPUT channel dominance = max_c E[c] / median_c E[c]."""
    ss = {n: None for n, _ in named_targets}
    cnt = {n: 0 for n, _ in named_targets}

    def mk(n):
        def hook(mod, inp, out, n=n):
            o = out.detach().reshape(-1, out.shape[-1]).float()
            s = (o * o).sum(0)
            ss[n] = s if ss[n] is None else ss[n] + s
            cnt[n] += o.shape[0]
        return hook
    h = [m.register_forward_hook(mk(n)) for n, m in named_targets]
    run_calib()
    for x in h:
        x.remove()
    out = {}
    for n in ss:
        e = (ss[n] / cnt[n]).cpu().numpy()
        out[n] = float(e.max() / np.median(e))
    return out


def main():
    tok = AutoTokenizer.from_pretrained(p4.MODEL)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    cal = tok("\n\n".join(t for t in ds["train"]["text"] if t.strip()),
              return_tensors="pt").input_ids.to(DEV)
    ev = tok("\n\n".join(t for t in ds["test"]["text"] if t.strip()),
             return_tensors="pt").input_ids.to(DEV)

    m0 = p4.fresh()
    tgts = list(p4.targets(m0))
    runc = lambda: [m0(cal[:, i * 2048:(i + 1) * 2048]) for i in range(8)]
    D = p5.collect_D(tgts, runc)
    outdom = collect_outdom(tgts, runc)
    del m0; torch.cuda.empty_cache()

    g = p4.group_of
    print("=== linkage: residual diagnostic (output dominance) vs D, by type ===")
    types = ["attn_in", "attn_out", "mlp_in", "mlp_out"]
    link = {}
    for t in types:
        ds_ = [outdom[n] for n, _ in tgts if g(n) == t]
        dd_ = [D[n] for n, _ in tgts if g(n) == t]
        link[t] = dict(out_dominance=round(float(np.median(ds_)), 1),
                       D=round(float(np.median(dd_)), 2))
        print(f"  {t:9s} out_dominance(median)={link[t]['out_dominance']:8.1f}  D(median)={link[t]['D']}")

    fp = round(p4.ppl(p4.fresh(), ev), 3)

    def go(spec):
        m = p4.fresh()
        ql = p4.wrap(m, spec)
        p4.calibrate(m, ql, cal)
        p = round(p4.ppl(m, ev), 3)
        del m; torch.cuda.empty_cache()
        return p

    thr = 5.0
    rbbqc = lambda n: dict(DYN) if (g(n) == "mlp_out" or D[n] >= thr) else dict(C1)
    n_dyn = sum(1 for n, _ in tgts if g(n) == "mlp_out" or D[n] >= thr)
    res = dict(fp=fp, n_total=len(tgts), n_dyn_rbbqc=n_dyn, thr=thr, linkage=link,
               plain=go(lambda n: dict(C1)),
               rbbqc=go(rbbqc),
               all_dynamic=go(lambda n: dict(DYN)))
    print(f"=== RBBQ-C (group-aware, thr={thr}) Qwen2.5-1.5B, FP={fp} ===")
    print(f"  plain={res['plain']}  rbbqc={res['rbbqc']} ({n_dyn}/{len(tgts)} dyn)  "
          f"all_dynamic={res['all_dynamic']}")
    json.dump(res, open("report/phase12_rbbqc.json", "w"), indent=2)
    print("wrote report/phase12_rbbqc.json")


if __name__ == "__main__":
    main()
