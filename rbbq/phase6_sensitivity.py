"""RBBQ pivot, Phase 6 — sensitivity-aware selection for selective-granularity W8A8.

Phase 5 selected per-token-dynamic linears by input spread
  D = max_token(rowmax)/median_token(rowmax).
Clean on Qwen, weak on BERT (ranked fc1 above the damaging fc2/o_proj). This phase
adds a direct, still-cheap sensitivity metric: the relative OUTPUT error that
static per-tensor activation quant induces at each linear,
  E = || q_static(x) W - x W || / || x W ||   (RMS over calibration tokens),
which accounts for the weight and the output norm, not just input spread. We select
the top-k linears by each metric at matched budgets k and compare accuracy/ppl.
"""
import json
import torch
import torch.nn.functional as F

import phase0_bert_sst2 as p0
import phase2_rbbq_a as p2
import phase3_variant_b as p3
import phase4_qwen_decomp as p4
import phase5_selective as p5
from phase3_variant_b import q_pt, FlexQuantLinear

DEV = "cuda"
STATIC, DYN = p5.STATIC, p5.DYN


def collect_maxabs(named_targets, run_calib):
    mx = {n: torch.tensor(0.0, device=DEV) for n, _ in named_targets}
    h = [m.register_forward_hook(
            lambda mod, inp, out, n=n: mx.__setitem__(
                n, torch.maximum(mx[n], inp[0].detach().abs().max().float())))
         for n, m in named_targets]
    run_calib()
    for x in h:
        x.remove()
    return {n: (v / 127.0).clamp_min(1e-8) for n, v in mx.items()}


def collect_E(named_targets, run_calib, scales):
    err = {n: [0.0, 0.0] for n, _ in named_targets}

    def mk(n, W):
        def hook(mod, inp, out, n=n, W=W):
            oq = F.linear(q_pt(inp[0].detach(), scales[n]), W)
            err[n][0] += ((oq - out.detach()).float() ** 2).sum().item()
            err[n][1] += (out.detach().float() ** 2).sum().item()
        return hook
    h = [m.register_forward_hook(mk(n, m.weight)) for n, m in named_targets]
    run_calib()
    for x in h:
        x.remove()
    return {n: (err[n][0] / max(err[n][1], 1e-12)) ** 0.5 for n in err}


def topk_set(metric, k):
    return set(sorted(metric, key=lambda n: -metric[n])[:k])


def run_bert(budgets):
    tok, ckpt, val_b, calib_b = p2.load_task("mnli")
    perm = p2.TASKS["mnli"]["perm"]
    fp = round(p2.run_method(ckpt, "fp", val_b, calib_b, perm=perm), 4)
    m0 = p2.fresh(ckpt)
    tgts = list(p0.iter_targets(m0))
    runc = lambda: [m0(**e) for e, _ in calib_b]
    D = p5.collect_D(tgts, runc)
    sc = collect_maxabs(tgts, runc)
    E = collect_E(tgts, runc, sc)
    del m0; torch.cuda.empty_cache()

    def go(dynset):
        m = p2.fresh(ckpt)
        ql = p5.wrap_generic(m, list(p0.iter_targets(m)),
                             lambda n: dict(DYN) if n in dynset else dict(STATIC))
        p3.calibrate(m, ql, calib_b)
        a = round(p2.evaluate(m, val_b, perm), 4)
        del m; torch.cuda.empty_cache()
        return a
    return assemble(fp, "acc", D, E, go, budgets, p5.bert_type, len(D))


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
    m0 = p4.fresh()
    tgts = list(p4.targets(m0))
    runc = lambda: [m0(cal[:, i * 2048:(i + 1) * 2048]) for i in range(8)]
    D = p5.collect_D(tgts, runc)
    sc = collect_maxabs(tgts, runc)
    E = collect_E(tgts, runc, sc)
    del m0; torch.cuda.empty_cache()

    def go(dynset):
        m = p4.fresh()
        ql = p5.wrap_generic(m, list(p4.targets(m)),
                             lambda n: dict(DYN) if n in dynset else dict(STATIC))
        p4.calibrate(m, ql, cal)
        p = round(p4.ppl(m, ev), 3)
        del m; torch.cuda.empty_cache()
        return p
    return assemble(fp, "ppl", D, E, go, budgets, p4.group_of, len(D))


def assemble(fp, metric_name, D, E, go, budgets, typef, ntot):
    res = dict(fp=fp, metric=metric_name, n_total=ntot,
               plain=go(set()), all_dynamic=go(set(D.keys())), budgets=[])
    for k in budgets:
        sd, se = topk_set(D, k), topk_set(E, k)
        res["budgets"].append(dict(k=k,
            D_select=go(sd), E_select=go(se),
            overlap=len(sd & se),
            D_types=p5.type_summary({n: 1 for n in sd}, typef),
            E_types=p5.type_summary({n: 1 for n in se}, typef)))
    return res


def main():
    out = {}
    out["bert"] = run_bert([18, 28])
    out["qwen"] = run_qwen([13, 32])
    for fam, r in out.items():
        print(f"=== {fam} ({r['metric']}) FP={r['fp']} plain={r['plain']} "
              f"all_dyn={r['all_dynamic']} /{r['n_total']} ===")
        for b in r["budgets"]:
            print(f"  k={b['k']:<3} D_select={b['D_select']}  E_select={b['E_select']}"
                  f"  overlap={b['overlap']}")
            print(f"       D picks {b['D_types']}")
            print(f"       E picks {b['E_types']}")
    json.dump(out, open("report/phase6_results.json", "w"), indent=2)
    print("wrote report/phase6_results.json")


if __name__ == "__main__":
    main()
