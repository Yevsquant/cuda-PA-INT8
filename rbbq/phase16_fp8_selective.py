"""RBBQ revised, Phase 16 — FP8 selective path (hardware-aware), the final component.

Solution #5: the D selection rule is format-agnostic. INT8 is the A100 path; FP8 is the
Hopper path (Phase 10: fp8 _scaled_mm = 0.6x bf16, while int8 GEMM is 4-7x). This phase
tests the FP8 (e4m3) ACCURACY question on Qwen2.5-1.5B (WikiText-2 ppl):
  - does FP8's wider dynamic range tolerate the down_proj per-token outliers that wreck
    INT8 static (110 ppl)? i.e. does FP8 even NEED selective granularity?
  - fp8 all-static | fp8 selective (dynamic at down_proj) | fp8 all-dynamic
vs the INT8 references (plain 110.4, RBBQ-C selective 10.49, all-dynamic 10.02).

Latency is settled by Phases 9-10: fp8 GEMM is faster than bf16 on Hopper and per-token
dynamic is free when fused, so the fp8 selective path is the Hopper latency win.
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import phase4_qwen_decomp as p4
from transformers import AutoTokenizer
from datasets import load_dataset

DEV = "cuda"
FP8_MAX = 448.0


def qfp8(x, scale):
    return (x / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(x.dtype) * scale


class Fp8Linear(nn.Module):
    def __init__(self, lin, dynamic=False):
        super().__init__()
        self.bias, self.dynamic = lin.bias, dynamic
        w = lin.weight.detach()
        sw = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / FP8_MAX     # per-channel weight
        self.register_buffer("wdq", qfp8(w, sw))
        self.register_buffer("act_scale", torch.tensor(0.0))
        self.calibrating = False

    def forward(self, x):
        if self.dynamic:
            s = (x.detach().abs().amax(-1, keepdim=True) / FP8_MAX).clamp_min(1e-8)
            return F.linear(qfp8(x, s), self.wdq, self.bias)
        if self.calibrating:
            self.act_scale = torch.maximum(self.act_scale, x.detach().abs().max().float())
            return F.linear(x, self.wdq, self.bias)
        s = (self.act_scale / FP8_MAX).clamp_min(1e-8)
        return F.linear(qfp8(x, s), self.wdq, self.bias)


def wrap_fp8(model, dyn_of):
    qlins = []
    for name, mod in list(p4.targets(model)):
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        child = name.rsplit(".", 1)[1]
        ql = Fp8Linear(mod, dynamic=dyn_of(name))
        setattr(parent, child, ql)
        qlins.append(ql)
    return qlins


@torch.no_grad()
def calib(model, qlins, cal):
    for q in qlins:
        if not q.dynamic:
            q.calibrating = True
    for i in range(8):
        model(cal[:, i * 2048:(i + 1) * 2048])
    for q in qlins:
        q.calibrating = False


def main():
    tok = AutoTokenizer.from_pretrained(p4.MODEL)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    cal = tok("\n\n".join(t for t in ds["train"]["text"] if t.strip()),
              return_tensors="pt").input_ids.to(DEV)
    ev = tok("\n\n".join(t for t in ds["test"]["text"] if t.strip()),
             return_tensors="pt").input_ids.to(DEV)
    fp = round(p4.ppl(p4.fresh(), ev), 3)
    g = p4.group_of

    def go(dyn_of):
        m = p4.fresh()
        ql = wrap_fp8(m, dyn_of)
        calib(m, ql, cal)
        p = round(p4.ppl(m, ev), 3)
        del m; torch.cuda.empty_cache()
        return p

    res = dict(fp=fp,
               fp8_all_static=go(lambda n: False),
               fp8_selective=go(lambda n: g(n) == "mlp_out"),
               fp8_all_dynamic=go(lambda n: True),
               int8_ref=dict(plain=110.378, rbbqc_selective=10.494, all_dynamic=10.022))
    print(f"=== Phase 16 FP8 (e4m3) on Qwen2.5-1.5B, FP ppl {fp} ===")
    for k in ["fp8_all_static", "fp8_selective", "fp8_all_dynamic"]:
        print(f"  {k:16s} ppl={res[k]:.3f} (dppl={res[k]-fp:+.3f})")
    print(f"  [int8 refs: plain {res['int8_ref']['plain']}, "
          f"RBBQ-C {res['int8_ref']['rbbqc_selective']}, all-dyn {res['int8_ref']['all_dynamic']}]")
    json.dump(res, open("report/phase16_fp8.json", "w"), indent=2)
    print("wrote report/phase16_fp8.json")


if __name__ == "__main__":
    main()
