"""RBBQ pivot, Phase 5 — Selective-granularity W8A8.

Phases 3-4 showed full static per-tensor W8A8 breaks at the MLP down-projection
input (activation per-token outliers), on both BERT and Qwen, while per-token
DYNAMIC act on just that linear recovers the FP16 oracle. This phase makes that a
*method*: an a-priori rule for WHICH linears need per-token dynamic, validated
against the per-linear damage, then evaluated vs plain-static and all-dynamic.

Selection statistic (cheap, from calibration): per linear input, the per-token
dynamic-range spread
    D = max_token(rowmax) / median_token(rowmax),   rowmax = max_c |x[t,c]|.
High D => one static per-tensor scale (set by the worst token) wastes most of the
int8 range on typical tokens => that linear needs per-token dynamic. The rule:
quantize a linear's activations per-token-dynamic iff D >= thr, else static.
"""
import json, math
import numpy as np
import torch
import torch.nn as nn

import phase0_bert_sst2 as p0
import phase2_rbbq_a as p2
import phase3_variant_b as p3
import phase4_qwen_decomp as p4
from phase3_variant_b import FlexQuantLinear

DEV = "cuda"
STATIC = dict(qw=True, qa=True, w_gran="tensor", act_mode="static")
DYN = dict(qw=True, qa=True, w_gran="channel", act_mode="dynamic")


def collect_D(named_targets, run_calib):
    store = {n: [] for n, _ in named_targets}
    handles = []
    for n, m in named_targets:
        def hook(mod, inp, out, n=n):
            store[n].append(inp[0].detach().abs().amax(-1).reshape(-1).float())
        handles.append(m.register_forward_hook(hook))
    run_calib()
    for h in handles:
        h.remove()
    D = {}
    for n, v in store.items():
        rm = torch.cat(v)
        D[n] = (rm.max() / rm.median().clamp_min(1e-9)).item()
    return D


def wrap_generic(model, named_targets, spec):
    qlins = []
    for name, mod in named_targets:
        opt = spec(name)
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], FlexQuantLinear(mod, **opt))
        qlins.append(model.get_submodule(name))
    return qlins


# ---------------- BERT ----------------
def bert_type(name):
    for t in ["query", "key", "value", "attention.output.dense",
              "intermediate.dense", "output.dense"]:
        if name.endswith(t):
            return {"query": "q", "key": "k", "value": "v",
                    "attention.output.dense": "o_proj",
                    "intermediate.dense": "fc1", "output.dense": "fc2"}[t]


def run_bert(thrs):
    tok, ckpt, val_b, calib_b = p2.load_task("mnli")
    perm = p2.TASKS["mnli"]["perm"]
    fp = p2.run_method(ckpt, "fp", val_b, calib_b, perm=perm)
    m0 = p2.fresh(ckpt)
    tgts = list(p0.iter_targets(m0))
    D = collect_D(tgts, lambda: [m0(**e) for e, _ in calib_b])
    del m0; torch.cuda.empty_cache()

    def go(spec):
        m = p2.fresh(ckpt)
        ql = wrap_generic(m, list(p0.iter_targets(m)), spec)
        p3.calibrate(m, ql, calib_b)
        a = round(p2.evaluate(m, val_b, perm), 4)
        del m; torch.cuda.empty_cache()
        return a

    res = dict(fp=round(fp, 4), plain=go(lambda n: dict(STATIC)),
               all_dynamic=go(lambda n: dict(DYN)), n_total=len(D), sweep=[])
    for thr in thrs:
        sel = lambda n, t=thr: dict(DYN) if D[n] >= t else dict(STATIC)
        res["sweep"].append(dict(thr=thr, n_dyn=int(sum(D[n] >= thr for n in D)),
                                 selective=go(sel)))
    return res, D, bert_type


# ---------------- Qwen ----------------
def run_qwen(thrs):
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(p4.MODEL)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    cal = tok("\n\n".join(t for t in ds["train"]["text"] if t.strip()),
              return_tensors="pt").input_ids.to(DEV)
    ev = tok("\n\n".join(t for t in ds["test"]["text"] if t.strip()),
             return_tensors="pt").input_ids.to(DEV)
    fp = p4.ppl(p4.fresh(), ev)
    m0 = p4.fresh()
    tgts = list(p4.targets(m0))
    D = collect_D(tgts, lambda: [m0(cal[:, i * 2048:(i + 1) * 2048]) for i in range(8)])
    del m0; torch.cuda.empty_cache()

    def go(spec):
        m = p4.fresh()
        ql = wrap_generic(m, list(p4.targets(m)), spec)
        p4.calibrate(m, ql, cal)
        p = round(p4.ppl(m, ev), 3)
        del m; torch.cuda.empty_cache()
        return p

    res = dict(fp=round(fp, 3), plain=go(lambda n: dict(STATIC)),
               all_dynamic=go(lambda n: dict(DYN)), n_total=len(D), sweep=[])
    for thr in thrs:
        sel = lambda n, t=thr: dict(DYN) if D[n] >= t else dict(STATIC)
        res["sweep"].append(dict(thr=thr, n_dyn=int(sum(D[n] >= thr for n in D)),
                                 selective=go(sel)))
    return res, D, (lambda n: p4.group_of(n))


def type_summary(D, typef):
    agg = {}
    for n, d in D.items():
        agg.setdefault(typef(n), []).append(d)
    return {t: round(float(np.median(v)), 2) for t, v in sorted(agg.items())}


def main():
    out = {}
    for fam, fn, thrs in [("bert", run_bert, [2.5, 3.0, 4.0, 6.0]),
                          ("qwen", run_qwen, [3.0, 4.0, 5.0, 8.0])]:
        res, D, typef = fn(thrs)
        res["D_by_type"] = type_summary(D, typef)
        out[fam] = res
        print(f"=== {fam} ===  FP={res['fp']}  plain={res['plain']}  "
              f"all_dynamic={res['all_dynamic']}  (of {res['n_total']} linears)")
        print("  D by linear type (median):", res["D_by_type"])
        for s in res["sweep"]:
            print(f"  thr={s['thr']:<4} n_dyn={s['n_dyn']:<3} selective={s['selective']}")
    json.dump(out, open("report/phase5_results.json", "w"), indent=2)
    print("wrote report/phase5_results.json")


if __name__ == "__main__":
    main()
