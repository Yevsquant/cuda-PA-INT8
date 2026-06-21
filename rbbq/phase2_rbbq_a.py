"""RBBQ Phase 2 — Variant A (exact, foldable boundary, branch-energy criterion).

Exact RBBQ-A can only act at the foldable LN->Linear boundary (qkv, fc1); per
Phase 1 the worst damage is at the MLP residual add (out_proj/fc2), which is NOT
LN-fed. This script measures, on C1 (static per-tensor W8A8), across SST-2 and
MNLI:

  fp        reference
  plain     W8A8, no smoothing
  sq        SmoothQuant smoothing at qkv/fc1   (magnitude criterion)
  rbbqA     RBBQ-A smoothing at qkv/fc1        (magnitude * branch-ratio^lambda)
  oracle_mp W8A8 but out_proj+fc2 kept FP16    (upper bound for foldable methods;
            isolates how much headroom lives in the non-foldable MLP-add region)

RBBQ-A scale:  s_c = a_c^0.5 / w_c^0.5 * clip(r[c], 0.1, 10)^lambda,
where r[c]=E_id/E_up is the branch imbalance of the residual add that PRODUCES
this linear's input stream (qkv_i <- mlp-add of layer i-1; fc1_i <- attn-add of
layer i). Hypothesis (Bondarenko): identity-dominated channels (r>1) are
persistent accumulated outliers and should be smoothed harder.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

import phase0_bert_sst2 as p0
import phase1_branch_energy as p1

DEV = "cuda"
EXCLUDE_MP = ("attention.output.dense", "output.dense")  # out_proj, fc2

TASKS = {
    "sst2": dict(ckpt="textattack/bert-base-uncased-SST-2", cfg="sst2",
                 val="validation", keys=("sentence", None), perm=None),
    # textattack MNLI has no id2label; output idx -> GLUE idx is (2,0,1) (84.6% FP)
    "mnli": dict(ckpt="textattack/bert-base-uncased-MNLI", cfg="mnli",
                 val="validation_matched", keys=("premise", "hypothesis"),
                 perm=(2, 0, 1)),
}


@torch.no_grad()
def evaluate(model, batches, perm=None):
    pmap = None if perm is None else torch.tensor(perm, device=DEV)
    correct = total = 0
    for enc, labels in batches:
        pred = model(**enc).logits.argmax(-1)
        if pmap is not None:
            pred = pmap[pred]
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / total


def make_batches(tok, rows, keys, bs=32, maxlen=128):
    k1, k2 = keys
    a, b = rows[k1], (rows[k2] if k2 else None)
    out = []
    for i in range(0, len(a), bs):
        args = (a[i:i + bs],) if b is None else (a[i:i + bs], b[i:i + bs])
        enc = tok(*args, padding=True, truncation=True, max_length=maxlen,
                  return_tensors="pt").to(DEV)
        out.append((enc, torch.tensor(rows["label"][i:i + bs]).to(DEV)))
    return out


def load_task(name, calib_n=256):
    t = TASKS[name]
    tok = AutoTokenizer.from_pretrained(t["ckpt"])
    ds = load_dataset("nyu-mll/glue", t["cfg"])
    val_b = make_batches(tok, ds[t["val"]][:], t["keys"])
    calib_b = make_batches(tok, ds["train"][:calib_n], t["keys"])
    return tok, t["ckpt"], val_b, calib_b


def fresh(ckpt):
    return AutoModelForSequenceClassification.from_pretrained(ckpt).to(DEV).eval()


@torch.no_grad()
def branch_energies(ckpt, calib_b):
    """Per-(layer,add) per-channel E_id, E_up via Phase 1 hooks on a clean model."""
    model = fresh(ckpt)
    stats = p1.BranchStats(model.config.hidden_size)
    p1.hook_bert(model, stats)
    for enc, _ in calib_b:
        p1.CUR_MASK = enc["attention_mask"].reshape(-1).bool()
        model(**enc)
    p1.CUR_MASK = None
    E = {}
    for (layer, add), sl in stats.s.items():
        E[(layer, add)] = (sl["ssq_id"] / sl["n"], sl["ssq_up"] / sl["n"])
    del model
    torch.cuda.empty_cache()
    return E


def layer_of(name):
    p = name.split(".")
    return int(p[p.index("layer") + 1])


def branch_r(name, E):
    """Imbalance r[c] for the input channels of a smoothed linear, or None."""
    i = layer_of(name)
    if name.endswith("intermediate.dense"):
        key = (i, 0)                       # fc1_i <- attn-add of layer i
    else:                                  # qkv_i <- mlp-add of layer i-1
        if i == 0:
            return None
        key = (i - 1, 1)
    if key not in E:
        return None
    e_id, e_up = E[key]
    return (e_id / e_up.clamp_min(1e-12))


@torch.no_grad()
def build_scales(model, calib_b, method, E=None, lam=0.5, alpha=0.5):
    pc = p0.collect_perchannel_max(model, calib_b)   # per-in-channel act max (qkv,fc1)
    scales = {}
    for name, mod in model.named_modules():
        if name in pc:
            a = pc[name].clamp_min(1e-5)
            w = mod.weight.detach().abs().amax(dim=0).clamp_min(1e-5)
            base = a.pow(alpha) / w.pow(1 - alpha)
            if method == "rbbqA" and E is not None:
                r = branch_r(name, E)
                if r is not None:
                    base = base * r.clamp(0.1, 10.0).pow(lam)
            scales[name] = base.clamp_min(1e-5)
    return scales


def quantize(model, scales=None, exclude=()):
    scales = scales or {}
    qlins = []
    for name, mod in list(p0.iter_targets(model)):
        if any(name.endswith(s) for s in exclude):
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        child = name.rsplit(".", 1)[1]
        ql = p0.QuantLinear(mod, "c1", smooth=scales.get(name))
        ql._orig_w = mod.weight.detach().clone()
        setattr(parent, child, ql)
        qlins.append(ql)
    return qlins


def run_method(ckpt, method, val_b, calib_b, E=None, lam=0.5, perm=None):
    model = fresh(ckpt)
    if method == "fp":
        return evaluate(model, val_b, perm)
    exclude = EXCLUDE_MP if method == "oracle_mp" else ()
    scales = None
    if method in ("sq", "rbbqA"):
        scales = build_scales(model, calib_b, method, E=E, lam=lam)
    qlins = quantize(model, scales=scales, exclude=exclude)
    p0.calibrate_c1(model, qlins, calib_b)
    acc = evaluate(model, val_b, perm)
    del model
    torch.cuda.empty_cache()
    return acc


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["sst2", "mnli"])
    ap.add_argument("--lam", type=float, default=0.5)
    args = ap.parse_args()

    methods = ["fp", "plain", "sq", "rbbqA", "oracle_mp"]
    out = {}
    for task in args.tasks:
        tok, ckpt, val_b, calib_b = load_task(task)
        perm = TASKS[task]["perm"]
        E = branch_energies(ckpt, calib_b)
        res = {}
        for m in methods:
            acc = run_method(ckpt, m, val_b, calib_b, E=E, lam=args.lam, perm=perm)
            res[m] = round(acc, 4)
            print(f"{task:5s} {m:10s} acc={acc:.4f}")
        res["_gap_plain"] = round(res["fp"] - res["plain"], 4)
        res["_gap_sq"] = round(res["fp"] - res["sq"], 4)
        res["_gap_rbbqA"] = round(res["fp"] - res["rbbqA"], 4)
        res["_gap_oracle"] = round(res["fp"] - res["oracle_mp"], 4)
        out[task] = res
        print(json.dumps(res, indent=2))

    out["_meta"] = dict(config="C1 static per-tensor W8A8", lam=args.lam)
    with open("report/phase2_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote report/phase2_results.json")


if __name__ == "__main__":
    main()
