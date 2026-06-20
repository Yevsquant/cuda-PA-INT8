"""Fake-quant decode-attention error on real vs synthetic K/V distributions.

Phase 1 verification: load the post-RoPE Q/K/V dumped by dump_kv.py, run a
single decode step (last query token attends over the whole sequence) with the
existing symmetric INT8 scheme in fake-quant (quant->dequant), and measure the
whole-vector output error (cosine + rel-L2 from Phase 0). For each real sample
we run a shape-matched N(0,1) control, so the table directly shows how much the
synthetic baseline *underestimates* real degradation — the audit's first point.

    python benchmarks/fakequant_eval.py --dump report/kv_dump.pt
"""

import argparse
import json
import math
import os
import sys

import torch

# Reuse the quant helpers + metrics that live under tests/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from paged_decode_attn import quantize_per_token, quantize_per_tensor  # noqa: E402
from quant_metrics import cosine_sim, rel_l2  # noqa: E402

MODES = ["fp16", "int8_per_token", "int8_per_tensor"]


def fake_quant(x, mode):
    """quant -> dequant round-trip of `x` ([..., head_dim]) under `mode`."""
    if mode == "fp16":
        return x
    if mode == "int8_per_token":
        q, s = quantize_per_token(x, dim=-1)
        return q.float() * s.unsqueeze(-1)
    if mode == "int8_per_tensor":
        q, s = quantize_per_tensor(x)
        return q.float() * s
    raise ValueError(mode)


def decode_attention(q_last, k, v, scale):
    """One decode step with GQA. q_last:[n_q,hd]; k,v:[n_kv,seq,hd] -> [n_q,hd]."""
    n_q, n_kv = q_last.shape[0], k.shape[0]
    group = n_q // n_kv
    k = k.repeat_interleave(group, dim=0)   # [n_q, seq, hd]
    v = v.repeat_interleave(group, dim=0)
    scores = torch.einsum("hd,hsd->hs", q_last, k) * scale   # [n_q, seq]
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("hs,hsd->hd", p, v)                  # [n_q, hd]


def k_outlier_ratio(k):
    """Per-channel amax over tokens; ratio of the largest channel to the median.
    >1 means a few channels dominate — the post-RoPE outlier-channel signature."""
    per_ch = k.abs().amax(dim=1)            # [n_kv, hd]
    return (per_ch.amax(dim=-1) / per_ch.median(dim=-1).values).mean().item()


def eval_sample(q, k, v):
    """Returns per-mode (cos_mean, rl2_mean, rl2_max) for one (q,k,v) in fp32."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    q_last = q[:, -1, :]                      # decode: the last query position
    ref = decode_attention(q_last, k, v, scale)
    out = {}
    for mode in MODES:
        o = decode_attention(q_last, fake_quant(k, mode), fake_quant(v, mode), scale)
        cos = cosine_sim(o, ref, dim=-1)
        rl2 = rel_l2(o, ref, dim=-1)
        out[mode] = (cos.mean().item(), rl2.mean().item(), rl2.max().item())
    return out


def aggregate(samples):
    """Mean over samples of each mode's (cos, rl2_mean, rl2_max)."""
    agg = {m: [0.0, 0.0, 0.0] for m in MODES}
    for s in samples:
        for m in MODES:
            for i in range(3):
                agg[m][i] += s[m][i] / len(samples)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="report/kv_dump.pt")
    ap.add_argument("--out", default="report/fakequant_eval.json")
    args = ap.parse_args()

    blob = torch.load(args.dump)
    meta, layers = blob["meta"], blob["layers"]
    torch.manual_seed(0)

    report = {"meta": meta, "per_layer": {}}
    print(f"model={meta['model']} n_q={meta['n_q_heads']} n_kv={meta['n_kv_heads']} "
          f"head_dim={meta['head_dim']}\n")
    hdr = f"{'layer':>5} {'dist':>9} {'K_outlier':>9} | " + " | ".join(
        f"{m:>14}" for m in MODES)
    print(hdr)
    print("-" * len(hdr))

    for idx in sorted(layers):
        real_samples, syn_samples, ratios = [], [], []
        for rec in layers[idx]:
            q, k, v = rec["q"].float(), rec["k"].float(), rec["v"].float()
            real_samples.append(eval_sample(q, k, v))
            ratios.append(k_outlier_ratio(k))
            # Shape-matched N(0,1) control.
            qs, ks, vs = torch.randn_like(q), torch.randn_like(k), torch.randn_like(v)
            syn_samples.append(eval_sample(qs, ks, vs))

        real, syn = aggregate(real_samples), aggregate(syn_samples)
        ratio = sum(ratios) / len(ratios)
        report["per_layer"][idx] = {
            "k_outlier_ratio": ratio, "real": real, "synthetic": syn}

        def fmt(agg):
            return " | ".join(
                f"c={agg[m][0]:.4f} L2={agg[m][1]:.4f}" for m in MODES)
        print(f"{idx:>5} {'real':>9} {ratio:>9.1f} | {fmt(real)}")
        print(f"{idx:>5} {'synthetic':>9} {'-':>9} | {fmt(syn)}")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {args.out}")
    print("(c = cosine mean, L2 = rel-L2 mean; higher c / lower L2 is better)")


if __name__ == "__main__":
    main()
