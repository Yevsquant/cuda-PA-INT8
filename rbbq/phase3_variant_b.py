"""RBBQ Phase 3 — Variant B (rebalance at the add) + decomposition diagnostic.

Phase 2 showed ~96% of the C1 W8A8 damage is at out_proj/fc2 (the non-foldable
MLP-add region). Variant B is a WEIGHT-side method: it folds a per-output-channel
scale into the update-branch output projection (W_down/out_proj) so that
per-tensor weight quant works, then applies a calibrated cross-norm inverse
(absorb into the following LayerNorm gamma; the post-norm wall makes this
approximate -> measured as eps_inv).

Step 1 (diagnostic, decides the ceiling): split the out_proj/fc2 damage into
  - weight-quant component  (quant weight, FP16 act)
  - activation-quant component (FP16 weight, quant act)
Variant B can only address the weight component.

Step 2 (Variant B): implemented if the weight component is material.
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import phase0_bert_sst2 as p0
import phase1_branch_energy as p1
import phase2_rbbq_a as p2

DEV = "cuda"
QMAX = 127.0
MP = ("attention.output.dense", "output.dense")  # out_proj, fc2 (update-branch outputs)


def q_pt(t, s):
    return torch.clamp(torch.round(t / s), -QMAX, QMAX) * s


class FlexQuantLinear(nn.Module):
    """C1-style fake quant with per-linear control of weight/act quant, plus
    Variant-B per-output-channel weight folding (out_fold) and its calibration."""
    def __init__(self, lin, qw=True, qa=True, w_gran="tensor", out_fold=None,
                 act_mode="static"):
        super().__init__()
        self.bias, self.qa, self.act_mode = lin.bias, qa, act_mode
        w = lin.weight.detach().clone()
        if out_fold is not None:                       # fold inverse into weight rows
            w = w / out_fold.to(w.dtype).unsqueeze(1)  # W_row_c /= d_c
        if qw:
            if w_gran == "tensor":
                s = w.abs().max().clamp_min(1e-8) / QMAX
            else:  # per output channel
                s = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / QMAX
            self.register_buffer("wq", q_pt(w, s))
        else:
            self.register_buffer("wq", w)
        self.register_buffer("act_scale", torch.tensor(0.0))
        self.calibrating = False
        self._orig_w = lin.weight.detach().clone()

    def forward(self, x):
        if not self.qa:
            return F.linear(x, self.wq, self.bias)
        if self.act_mode == "dynamic":  # per-token, no calibration
            s = (x.detach().abs().amax(-1, keepdim=True) / QMAX).clamp_min(1e-8)
            return F.linear(q_pt(x, s), self.wq, self.bias)
        if self.calibrating:
            self.act_scale = torch.maximum(self.act_scale, x.detach().abs().max())
            return F.linear(x, self._orig_w, self.bias)
        xq = q_pt(x, (self.act_scale / QMAX).clamp_min(1e-8))
        return F.linear(xq, self.wq, self.bias)


def wrap(model, spec):
    """spec(name) -> dict(qw,qa,w_gran,out_fold) or None to skip (leave FP)."""
    qlins = []
    for name, mod in list(p0.iter_targets(model)):
        opt = spec(name)
        if opt is None:
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        child = name.rsplit(".", 1)[1]
        ql = FlexQuantLinear(mod, **opt)
        setattr(parent, child, ql)
        qlins.append(ql)
    return qlins


@torch.no_grad()
def calibrate(model, qlins, calib_b):
    for ql in qlins:
        if ql.qa:
            ql.calibrating = True
    for enc, _ in calib_b:
        model(**enc)
    for ql in qlins:
        ql.calibrating = False


def is_mp(name):
    return any(name.endswith(s) for s in MP)


# ---- Step 1: decomposition diagnostic ----
def run_spec(ckpt, spec, val_b, calib_b, perm):
    model = p2.fresh(ckpt)
    qlins = wrap(model, spec)
    calibrate(model, qlins, calib_b)
    acc = p2.evaluate(model, val_b, perm)
    del model; torch.cuda.empty_cache()
    return acc


def diagnostic(ckpt, val_b, calib_b, perm):
    base = dict(qw=True, qa=True, w_gran="tensor")
    specs = {
        # everything quantized (C1)
        "plain":   lambda n: dict(base),
        # out_proj/fc2 fully FP16 (oracle)
        "oracle":  lambda n: None if is_mp(n) else dict(base),
        # out_proj/fc2: FP16 weight, quant act -> isolates ACTIVATION damage
        "diag_w":  lambda n: dict(qw=False, qa=True, w_gran="tensor") if is_mp(n) else dict(base),
        # out_proj/fc2: quant weight (per-tensor), FP16 act -> isolates WEIGHT damage
        "diag_a":  lambda n: dict(qw=True, qa=False, w_gran="tensor") if is_mp(n) else dict(base),
        # out_proj/fc2: PER-CHANNEL weight, quant act -> what perfect weight handling gives
        "diag_wc": lambda n: dict(qw=True, qa=True, w_gran="channel") if is_mp(n) else dict(base),
        # CONSTRUCTIVE FIX: out_proj/fc2 use per-token DYNAMIC act + per-channel weight,
        # rest stays C1 static -> tests whether the damage is a granularity problem
        "fix_dyn": lambda n: (dict(qw=True, qa=True, w_gran="channel", act_mode="dynamic")
                              if is_mp(n) else dict(base)),
    }
    out = {}
    for k, sp in specs.items():
        out[k] = round(run_spec(ckpt, sp, val_b, calib_b, perm), 4)
        print(f"  {k:8s} acc={out[k]:.4f}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mnli")
    args = ap.parse_args()
    tok, ckpt, val_b, calib_b = p2.load_task(args.task)
    perm = p2.TASKS[args.task]["perm"]
    fp = p2.run_method(ckpt, "fp", val_b, calib_b, perm=perm)
    print(f"=== {args.task} decomposition (FP={fp:.4f}) ===")
    d = diagnostic(ckpt, val_b, calib_b, perm)
    res = {"fp": round(fp, 4), **d,
           "gap": {k: round(fp - v, 4) for k, v in d.items()}}
    print(json.dumps(res["gap"], indent=2))
    json.dump(res, open(f"report/phase3_diag_{args.task}.json", "w"), indent=2)
    print(f"wrote report/phase3_diag_{args.task}.json")


if __name__ == "__main__":
    main()
