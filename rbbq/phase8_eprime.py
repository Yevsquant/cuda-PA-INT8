"""RBBQ pivot, Phase 8 — E' selection refinement.

Phase 6: input-spread D beats a naive sensitivity metric E = ||q_static(x)W - xW||/||xW||
(local static-vs-FP error), because E conflates "hard to quantize locally" with
"matters end-to-end". The principled output-space metric is the error that per-token
dynamic *removes* at each linear:
    E' = || q_static(x) W - q_dynamic(x) W || / || x W ||.
E' is the output-space analog of D (which measures the static-vs-dynamic input scale
mismatch). This phase compares D, E, E' selection at matched budgets on both families.
"""
import json
import torch
import torch.nn.functional as F

import phase0_bert_sst2 as p0
import phase2_rbbq_a as p2
import phase3_variant_b as p3
import phase4_qwen_decomp as p4
import phase5_selective as p5
import phase6_sensitivity as p6
from phase3_variant_b import q_pt

DEV = "cuda"
STATIC, DYN = p5.STATIC, p5.DYN


def collect_Eprime(named_targets, run_calib, scales):
    """Output error that per-token dynamic removes vs static, per linear."""
    err = {n: [0.0, 0.0] for n, _ in named_targets}

    def mk(n, W):
        def hook(mod, inp, out, n=n, W=W):
            x = inp[0].detach()
            os = F.linear(q_pt(x, scales[n]), W)
            sd = (x.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-8)
            od = F.linear(q_pt(x, sd), W)
            err[n][0] += ((os - od).float() ** 2).sum().item()
            err[n][1] += (out.detach().float() ** 2).sum().item()
        return hook
    h = [m.register_forward_hook(mk(n, m.weight)) for n, m in named_targets]
    run_calib()
    for x in h:
        x.remove()
    return {n: (err[n][0] / max(err[n][1], 1e-12)) ** 0.5 for n in err}


def compare(fp, metric_name, metrics, go, budgets, typef, ntot):
    res = dict(fp=fp, metric=metric_name, n_total=ntot,
               plain=go(set()), all_dynamic=go(set(next(iter(metrics.values())).keys())),
               budgets=[])
    for k in budgets:
        row = dict(k=k)
        for mname, mvals in metrics.items():
            sk = p6.topk_set(mvals, k)
            row[mname] = go(sk)
            row[mname + "_types"] = p5.type_summary({n: 1 for n in sk}, typef)
        res["budgets"].append(row)
    return res


def run_bert(budgets):
    tok, ckpt, val_b, calib_b = p2.load_task("mnli")
    perm = p2.TASKS["mnli"]["perm"]
    fp = round(p2.run_method(ckpt, "fp", val_b, calib_b, perm=perm), 4)
    m0 = p2.fresh(ckpt); tgts = list(p0.iter_targets(m0))
    runc = lambda: [m0(**e) for e, _ in calib_b]
    sc = p6.collect_maxabs(tgts, runc)
    metrics = dict(D=p5.collect_D(tgts, runc), E=p6.collect_E(tgts, runc, sc),
                   Ep=collect_Eprime(tgts, runc, sc))
    del m0; torch.cuda.empty_cache()

    def go(dynset):
        m = p2.fresh(ckpt)
        ql = p5.wrap_generic(m, list(p0.iter_targets(m)),
                             lambda n: dict(DYN) if n in dynset else dict(STATIC))
        p3.calibrate(m, ql, calib_b)
        a = round(p2.evaluate(m, val_b, perm), 4)
        del m; torch.cuda.empty_cache()
        return a
    return compare(fp, "acc", metrics, go, budgets, p5.bert_type, len(tgts))


def run_qwen(budgets):
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(p4.MODEL)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    cal = tok("\n\n".join(t for t in ds["train"]["text"] if t.strip()),
              return_tensors="pt").input_ids.to(DEV)
    ev = tok("\n\n".join(t for t in ds["test"]["text"] if t.strip()),
             return_tensors="pt").input_ids.to(DEV)
    fp = round(p4.ppl(p4.fresh(), ev), 3)
    m0 = p4.fresh(); tgts = list(p4.targets(m0))
    runc = lambda: [m0(cal[:, i * 2048:(i + 1) * 2048]) for i in range(8)]
    sc = p6.collect_maxabs(tgts, runc)
    metrics = dict(D=p5.collect_D(tgts, runc), E=p6.collect_E(tgts, runc, sc),
                   Ep=collect_Eprime(tgts, runc, sc))
    del m0; torch.cuda.empty_cache()

    def go(dynset):
        m = p4.fresh()
        ql = p5.wrap_generic(m, list(p4.targets(m)),
                             lambda n: dict(DYN) if n in dynset else dict(STATIC))
        p4.calibrate(m, ql, cal)
        p = round(p4.ppl(m, ev), 3)
        del m; torch.cuda.empty_cache()
        return p
    return compare(fp, "ppl", metrics, go, budgets, p4.group_of, len(tgts))


def main():
    out = {"bert": run_bert([18, 28]), "qwen": run_qwen([13, 32])}
    for fam, r in out.items():
        print(f"=== {fam} ({r['metric']}) FP={r['fp']} plain={r['plain']} "
              f"all_dyn={r['all_dynamic']} /{r['n_total']} ===")
        for b in r["budgets"]:
            print(f"  k={b['k']:<3} D={b['D']}  E={b['E']}  E'={b['Ep']}")
            print(f"       E' picks {b['Ep_types']}")
    json.dump(out, open("report/phase8_results.json", "w"), indent=2)
    print("wrote report/phase8_results.json")


if __name__ == "__main__":
    main()
