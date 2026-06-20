"""End-to-end WikiText-2 perplexity with fake-quant KV (Phase 5).

Translates the kernel-level rel-L2 numbers into perplexity — the metric reviewers
actually weigh. A hook on eager_attention_forward replaces post-RoPE K (and V)
with quant->dequant versions under each mode, so the model runs as if it had an
INT8 KV cache. SmoothQuant/Hadamard transform Q,K first (exact on the scores);
all INT8 modes quantize V per-token symmetric, matching the kernel.

    python benchmarks/ppl_eval.py --max-tokens 60000

bf16, NOT fp16: Qwen2.5 deep-layer activations overflow fp16 to NaN.
"""

import argparse
import json
import math
import os
import sys

import torch
import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from quant_transforms import hadamard_matrix, smoothquant_factor  # noqa: E402

MODES = ["baseline", "int8_per_tensor", "int8_per_token",
         "int8_smoothquant", "int8_asym", "int8_hadamard", "fp8_e4m3"]


def _fq_per_token(x):
    s = (x.abs().amax(-1, keepdim=True) / 127.0).clamp_min(1e-8)
    return (torch.round(x / s).clamp(-127, 127) * s).to(x.dtype)


def _fq_fp8(x):
    """Per-token E4M3 round-trip (vLLM's only native sub-fp16 KV option). A true
    float8_e4m3fn cast, scaled so each token's max maps to E4M3's 448 limit."""
    s = (x.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-8)
    # clamp before the cast: rounding can nudge a scaled value just past 448,
    # which becomes NaN in e4m3fn (no inf) and poisons the output.
    q = (x / s).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (q * s).to(x.dtype)


def _fq_per_tensor(x):
    s = (x.abs().amax() / 127.0).clamp_min(1e-8)
    return (torch.round(x / s).clamp(-127, 127) * s).to(x.dtype)


def _fq_asym(x):
    xmax = x.amax(-1, keepdim=True)
    xmin = x.amin(-1, keepdim=True)
    s = ((xmax - xmin) / 255.0).clamp_min(1e-8)
    z = torch.round(-128.0 - xmin / s)
    return ((torch.round(x / s + z).clamp(-128, 127) - z) * s).to(x.dtype)


def make_hook(orig, mode, alpha, H):
    """Wrap eager_attention_forward to fake-quant K/V (and pre-transform Q,K)."""
    def hook(module, query, key, value, *a, **kw):
        q, k, v = query.float(), key.float(), value.float()
        n_q, n_kv = q.shape[1], k.shape[1]
        group = n_q // n_kv

        if mode == "int8_smoothquant":
            k_amax = k.abs().amax(dim=(0, 2))                                   # [n_kv,hd]
            q_amax = q.view(q.shape[0], n_kv, group, q.shape[2], q.shape[3]).abs().amax(dim=(0, 2, 3))
            s = smoothquant_factor(k_amax, q_amax, alpha=alpha)                 # [n_kv,hd]
            k = k / s[None, :, None, :]
            q = q * s.repeat_interleave(group, 0)[None, :, None, :]
        elif mode == "int8_hadamard":
            Hd = H.to(q.dtype)
            q, k = q @ Hd, k @ Hd

        if mode == "int8_per_tensor":
            k, v = _fq_per_tensor(k), _fq_per_tensor(v)
        elif mode == "int8_asym":
            k, v = _fq_asym(k), _fq_per_token(v)   # asym K only; V symmetric
        elif mode == "fp8_e4m3":
            k, v = _fq_fp8(k), _fq_fp8(v)
        else:  # per_token / smoothquant / hadamard
            k, v = _fq_per_token(k), _fq_per_token(v)

        return orig(module, q.to(query.dtype), k.to(key.dtype), v.to(value.dtype), *a, **kw)
    return hook


def perplexity(model, ids, device, max_len=2048, stride=1024):
    """Sliding-window causal-LM perplexity over a 1-D token tensor `ids`."""
    nll_sum, n_tok, prev = 0.0, 0, 0
    for begin in range(0, ids.numel(), stride):
        end = min(begin + max_len, ids.numel())
        trg = end - prev
        window = ids[begin:end].unsqueeze(0).to(device)
        target = window.clone()
        target[:, :-trg] = -100
        with torch.no_grad():
            loss = model(window, labels=target).loss
        nll_sum += loss.item() * trg
        n_tok += trg
        prev = end
        if end == ids.numel():
            break
    return math.exp(nll_sum / n_tok), n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="report/ppl_eval.json")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--max-tokens", type=int, default=60000)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][:args.max_tokens]

    H = hadamard_matrix(model.config.hidden_size // model.config.num_attention_heads,
                        device=device)
    orig = qwen2_mod.eager_attention_forward

    report = {"model": args.model, "alpha": args.alpha,
              "n_tokens": int(ids.numel()), "ppl": {}}
    print(f"model={args.model}  tokens={ids.numel()}  (lower PPL is better)\n")
    print(f"{'mode':>18} {'PPL':>9} {'dPPL%':>8}")
    print("-" * 38)

    base = None
    for mode in MODES:
        qwen2_mod.eager_attention_forward = (
            orig if mode == "baseline" else make_hook(orig, mode, args.alpha, H))
        try:
            ppl, _ = perplexity(model, ids, device, args.max_len, args.stride)
        finally:
            qwen2_mod.eager_attention_forward = orig
        report["ppl"][mode] = ppl
        if base is None:
            base = ppl
        d = 100.0 * (ppl / base - 1.0)
        print(f"{mode:>18} {ppl:>9.3f} {d:>+7.1f}%")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
