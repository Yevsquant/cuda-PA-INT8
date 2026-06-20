"""RBBQ Phase 0 — baseline harness on BERT-base / SST-2.

Stands up the W8A8 fake-quant + SmoothQuant pipeline that Phases 2-3 generalize,
and reproduces the FP accuracy of a known fine-tuned checkpoint as the gate.

Configs evaluated:
  fp         : reference (fp32 on GPU)
  c1         : static per-tensor act + per-tensor weight  (the HARD setting)
  c1_sq      : c1 + SmoothQuant smoothing (qkv, fc1)
  c2         : per-token dynamic act + per-channel weight  (the EASY setting)
  c2_sq      : c2 + SmoothQuant smoothing

Smoothing note (post-norm BERT): every LayerNorm output is the residual stream,
so the smooth scale is NOT folded into the LN (that would break the residual add)
-- it is applied as an explicit, exact per-channel pre-linear scale: x/s @ (W*s).
Accuracy is identical to folded SmoothQuant; foldability is a Phase 5 concern.
"""
import argparse, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

MODEL = "textattack/bert-base-uncased-SST-2"
DEV = "cuda"
QMAX = 127.0  # symmetric int8

# linears we quantize (per encoder layer, relative names) + which get smoothed
QUANT_SUFFIXES = [
    "attention.self.query", "attention.self.key", "attention.self.value",
    "attention.output.dense", "intermediate.dense", "output.dense",
]
SMOOTH_SUFFIXES = [
    "attention.self.query", "attention.self.key", "attention.self.value",
    "intermediate.dense",
]


def quant_per_tensor(t, scale):
    return torch.clamp(torch.round(t / scale), -QMAX, QMAX) * scale


def wscale_per_tensor(w):
    return w.abs().max().clamp_min(1e-8) / QMAX


def wscale_per_channel(w):  # per output row
    return w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / QMAX


class QuantLinear(nn.Module):
    def __init__(self, lin, mode, smooth=None):
        super().__init__()
        self.mode = mode
        self.bias = lin.bias
        s = None if smooth is None else smooth.to(lin.weight.dtype)
        self.register_buffer("smooth", s, persistent=False)
        w = lin.weight.detach().clone()
        if s is not None:
            w = w * s  # W * s  (per input channel)
        if mode == "c1":
            self.register_buffer("wq", quant_per_tensor(w, wscale_per_tensor(w)))
        else:  # c2: per-channel weight
            self.register_buffer("wq", quant_per_tensor(w, wscale_per_channel(w)))
        self.register_buffer("act_scale", torch.tensor(0.0))  # c1 static, filled by calib
        self.calibrating = False

    def forward(self, x):
        xs = x if self.smooth is None else x / self.smooth
        if self.mode == "c1":
            if self.calibrating:
                m = xs.detach().abs().max()
                self.act_scale = torch.maximum(self.act_scale, m)
                return F.linear(x if self.smooth is None else xs * self.smooth,
                                self._orig_w, self.bias)  # exact passthrough
            xq = quant_per_tensor(xs, (self.act_scale / QMAX).clamp_min(1e-8))
        else:  # c2 dynamic per-token
            sc = (xs.detach().abs().amax(dim=-1, keepdim=True) / QMAX).clamp_min(1e-8)
            xq = quant_per_tensor(xs, sc)
        return F.linear(xq, self.wq, self.bias)


def iter_targets(model):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(name.endswith(s) for s in QUANT_SUFFIXES):
            yield name, mod


def collect_perchannel_max(model, batches):
    """Per-input-channel max |act| at each smoothed linear's input."""
    stats, handles = {}, []

    def mk(name):
        def hook(mod, inp, out):
            a = inp[0].detach().abs().amax(dim=tuple(range(inp[0].dim() - 1)))
            stats[name] = torch.maximum(stats[name], a) if name in stats else a
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(name.endswith(s) for s in SMOOTH_SUFFIXES):
            handles.append(mod.register_forward_hook(mk(name)))
    run_batches(model, batches)
    for h in handles:
        h.remove()
    return stats


def smooth_scales(model, pc_max, alpha=0.5):
    out = {}
    for name, mod in model.named_modules():
        if name in pc_max:
            a = pc_max[name].clamp_min(1e-5)
            w = mod.weight.detach().abs().amax(dim=0).clamp_min(1e-5)  # per in-channel
            out[name] = (a.pow(alpha) / w.pow(1 - alpha)).clamp_min(1e-5)
    return out


def quantize_model(model, mode, smooth=False, pc_max=None):
    scales = smooth_scales(model, pc_max) if smooth else {}
    qlins = []
    for name, mod in list(iter_targets(model)):
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        child = name.rsplit(".", 1)[1]
        ql = QuantLinear(mod, mode, smooth=scales.get(name))
        ql._orig_w = mod.weight.detach().clone()  # for c1 calibration passthrough
        setattr(parent, child, ql)
        qlins.append(ql)
    return qlins


def calibrate_c1(model, qlins, batches):
    for ql in qlins:
        ql.calibrating = True
    run_batches(model, batches)
    for ql in qlins:
        ql.calibrating = False
        del ql._orig_w


# ---- data / eval ----
def make_batches(tok, rows, bs=32, maxlen=128):
    out = []
    for i in range(0, len(rows["sentence"]), bs):
        enc = tok(rows["sentence"][i:i + bs], padding=True, truncation=True,
                  max_length=maxlen, return_tensors="pt")
        labels = torch.tensor(rows["label"][i:i + bs])
        out.append(({k: v.to(DEV) for k, v in enc.items()}, labels.to(DEV)))
    return out


@torch.no_grad()
def run_batches(model, batches):
    for enc, _ in batches:
        model(**enc)


@torch.no_grad()
def evaluate(model, batches):
    correct = total = 0
    for enc, labels in batches:
        pred = model(**enc).logits.argmax(-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["fp", "c1", "c1_sq", "c2", "c2_sq"])
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--out", default="report/phase0_results.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    val = load_dataset("nyu-mll/glue", "sst2")["validation"][:]
    train = load_dataset("nyu-mll/glue", "sst2")["train"][:args.calib]
    val_b = make_batches(tok, val)
    calib_b = make_batches(tok, train)

    results = {}
    for cfg in args.configs:
        t0 = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(DEV).eval()
        if cfg == "fp":
            acc = evaluate(model, val_b)
        else:
            mode = "c1" if cfg.startswith("c1") else "c2"
            smooth = cfg.endswith("_sq")
            pc = collect_perchannel_max(model, calib_b) if smooth else None
            qlins = quantize_model(model, mode, smooth=smooth, pc_max=pc)
            if mode == "c1":
                calibrate_c1(model, qlins, calib_b)
            acc = evaluate(model, val_b)
        results[cfg] = round(acc, 4)
        print(f"{cfg:8s} acc={acc:.4f}  ({time.time()-t0:.1f}s)")
        del model
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
