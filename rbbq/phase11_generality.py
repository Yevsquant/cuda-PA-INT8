"""RBBQ pivot, Phase 11 — generality on a different family at 7B (Mistral-7B).

Tests whether the Phase 3-5 conclusions transfer beyond BERT/Qwen:
  (1) does W8A8 damage localize to the MLP down-projection?
  (2) is it activation-side (not weight)?
  (3) does per-token dynamic / the D-selection rule fix it?
on Mistral-7B-Instruct-v0.2 (different family from Qwen; Llama-style pre-norm RMSNorm
+ SwiGLU; same {q,k,v,o}_proj / {gate,up,down}_proj layout, so the Phase 4 harness
reuses verbatim). Metric: WikiText-2 perplexity, C1 static per-tensor W8A8.
"""
import json
import torch

import phase4_qwen_decomp as p4
import phase5_selective as p5
import phase6_sensitivity as p6
from transformers import AutoTokenizer
from datasets import load_dataset

p4.MODEL = "mistralai/Mistral-7B-Instruct-v0.2"   # reuse p4.fresh/targets/ppl/calibrate
DEV = "cuda"
C1 = dict(qw=True, qa=True, w_gran="tensor")
DYN = p5.DYN


def main():
    tok = AutoTokenizer.from_pretrained(p4.MODEL)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    cal = tok("\n\n".join(t for t in ds["train"]["text"] if t.strip()),
              return_tensors="pt").input_ids.to(DEV)
    ev = tok("\n\n".join(t for t in ds["test"]["text"] if t.strip()),
             return_tensors="pt").input_ids.to(DEV)

    def ppl_of(model):
        return p4.ppl(model, ev, seqlen=2048, n=20)

    fp = round(ppl_of(p4.fresh()), 3)
    print(f"=== Mistral-7B WikiText-2 ppl (FP bf16 = {fp}) ===")

    # D statistic on a clean model
    m0 = p4.fresh()
    tgts = list(p4.targets(m0))
    D = p5.collect_D(tgts, lambda: [m0(cal[:, i * 2048:(i + 1) * 2048]) for i in range(8)])
    del m0; torch.cuda.empty_cache()
    n_down = sum(1 for n, _ in tgts if n.endswith("down_proj"))
    Dtop = p6.topk_set(D, n_down)                       # top-(n_down) linears by D
    print("D by type (median):", p5.type_summary(D, p4.group_of))
    print("D-top types:", p5.type_summary({n: 1 for n in Dtop}, p4.group_of))

    def go(spec):
        m = p4.fresh()
        ql = p4.wrap(m, spec)
        p4.calibrate(m, ql, cal)
        p = round(ppl_of(m), 3)
        del m; torch.cuda.empty_cache()
        return p

    g = p4.group_of
    arms = {
        "plain":       lambda n: dict(C1),
        "fp16_mlp_out": lambda n: None if g(n) == "mlp_out" else dict(C1),
        "mlp_out_w":   lambda n: dict(qw=False, qa=True, w_gran="tensor") if g(n) == "mlp_out" else dict(C1),
        "mlp_out_a":   lambda n: dict(qw=True, qa=False, w_gran="tensor") if g(n) == "mlp_out" else dict(C1),
        "mlp_out_dyn": lambda n: dict(DYN) if g(n) == "mlp_out" else dict(C1),
        "D_selective": lambda n: dict(DYN) if n in Dtop else dict(C1),
        "all_dynamic": lambda n: dict(DYN),
    }
    res = {"fp": fp, "n_total": len(tgts), "n_down": n_down,
           "D_by_type": p5.type_summary(D, g),
           "D_top_types": p5.type_summary({n: 1 for n in Dtop}, g)}
    for name, sp in arms.items():
        res[name] = go(sp)
        print(f"  {name:13s} ppl={res[name]:.3f}  (dppl={res[name]-fp:+.3f})")
    json.dump(res, open("report/phase11_mistral.json", "w"), indent=2)
    print("wrote report/phase11_mistral.json")


if __name__ == "__main__":
    main()
