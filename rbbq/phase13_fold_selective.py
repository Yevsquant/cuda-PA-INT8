"""RBBQ revised, Phase 13 — RBBQ-A + selective ablation (exact folds where legal).

Solution #2: keep exact LN->Linear folding (SmoothQuant/RBBQ-A) on the FOLDABLE linears
only, and per-token dynamic at down_proj. In pre-norm Qwen the RMSNorm output feeds only
qkv / gate-up (NOT the residual), so folding into the RMSNorm weight is exact (unlike
post-norm BERT). Ablation on Qwen2.5-1.5B (WikiText-2 ppl, C1 W8A8):
  plain | folds_only | selective_only | folds+selective | all_dynamic
Question: does legal folding add anything on top of selective dynamic at down_proj?
"""
import json
import torch

import phase4_qwen_decomp as p4
import phase5_selective as p5
from transformers import AutoTokenizer
from datasets import load_dataset

DEV = "cuda"
C1 = dict(qw=True, qa=True, w_gran="tensor")
DYN = p5.DYN


@torch.no_grad()
def smooth_qwen(model, cal, alpha=0.5):
    """Exact SmoothQuant folding on qkv (input_layernorm) and gate/up
    (post_attention_layernorm). Pre-norm => folding into RMSNorm weight is exact."""
    layers = model.model.layers
    groups = []  # (ln, [linears])
    for L in layers:
        groups.append((L.input_layernorm, [L.self_attn.q_proj, L.self_attn.k_proj, L.self_attn.v_proj]))
        groups.append((L.post_attention_layernorm, [L.mlp.gate_proj, L.mlp.up_proj]))
    amax = {id(g[0]): None for g in groups}
    handles = []
    for ln, lins in groups:
        handles.append(lins[0].register_forward_pre_hook(
            lambda m, a, key=id(ln): _store(amax, key, a[0])))
    for i in range(8):
        model(cal[:, i * 2048:(i + 1) * 2048])
    for h in handles:
        h.remove()
    for ln, lins in groups:
        a = amax[id(ln)].clamp_min(1e-5)
        w = torch.stack([l.weight.detach().abs().amax(0) for l in lins]).amax(0).clamp_min(1e-5)
        s = (a.pow(alpha) / w.pow(1 - alpha)).clamp_min(1e-5).to(ln.weight.dtype)
        ln.weight.div_(s)
        for l in lins:
            l.weight.mul_(s.to(l.weight.dtype))


def _store(d, key, x):
    a = x.detach().abs().amax(dim=tuple(range(x.dim() - 1))).float()
    d[key] = a if d[key] is None else torch.maximum(d[key], a)


def main():
    tok = AutoTokenizer.from_pretrained(p4.MODEL)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    cal = tok("\n\n".join(t for t in ds["train"]["text"] if t.strip()),
              return_tensors="pt").input_ids.to(DEV)
    ev = tok("\n\n".join(t for t in ds["test"]["text"] if t.strip()),
             return_tensors="pt").input_ids.to(DEV)
    fp = round(p4.ppl(p4.fresh(), ev), 3)

    g = p4.group_of
    sel = lambda n: dict(DYN) if g(n) == "mlp_out" else dict(C1)

    def go(spec, smooth):
        m = p4.fresh()
        if smooth:
            smooth_qwen(m, cal)
        ql = p4.wrap(m, spec)
        p4.calibrate(m, ql, cal)
        p = round(p4.ppl(m, ev), 3)
        del m; torch.cuda.empty_cache()
        return p

    res = dict(fp=fp,
               plain=go(lambda n: dict(C1), False),
               folds_only=go(lambda n: dict(C1), True),
               selective_only=go(sel, False),
               folds_selective=go(sel, True),
               all_dynamic=go(lambda n: dict(DYN), False))
    print(f"=== Phase 13 ablation Qwen2.5-1.5B (FP ppl {fp}) ===")
    for k in ["plain", "folds_only", "selective_only", "folds_selective", "all_dynamic"]:
        print(f"  {k:16s} ppl={res[k]:.3f}  (dppl={res[k]-fp:+.3f})")
    json.dump(res, open("report/phase13_ablation.json", "w"), indent=2)
    print("wrote report/phase13_ablation.json")


if __name__ == "__main__":
    main()
