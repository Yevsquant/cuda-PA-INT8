"""SmoothQuant (方案 B) on real post-RoPE K/V: does it beat plain symmetric INT8?

Phase 2 verification. For each dumped layer we calibrate a per-(kv_head, channel)
SmoothQuant factor from the layer's tokens, sweep alpha, and compare the decode
rel-L2 of `int8_per_token` vs `smoothquant + int8_per_token`. Reference is the
raw-fp output; V is quantized identically in both arms, so the only difference is
K smoothing. We also assert Q'·K'ᵀ == Q·Kᵀ (the transform is free in fp).

    python benchmarks/smoothquant_eval.py --dump report/kv_dump.pt
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from quant_metrics import rel_l2  # noqa: E402
from quant_transforms import apply_smooth, smoothquant_factor  # noqa: E402
from fakequant_eval import decode_attention, fake_quant  # noqa: E402

ALPHAS = [0.3, 0.5, 0.7, 0.85, 1.0]


def calibrate(samples, n_q, n_kv):
    """Per-(kv_head, channel) abs-max over all tokens of all samples.
    K at kv-head granularity; Q reduced over the q-heads in each GQA group."""
    group = n_q // n_kv
    k_amax = torch.zeros(n_kv, samples[0]["k"].shape[-1])
    q_amax = torch.zeros(n_kv, samples[0]["k"].shape[-1])
    for rec in samples:
        k = rec["k"].float()                       # [n_kv, seq, hd]
        k_amax = torch.maximum(k_amax, k.abs().amax(dim=1))
        # q: [n_q, seq, hd] -> group by kv head -> max over (group, seq)
        qg = rec["q"].float().view(n_kv, group, -1, rec["q"].shape[-1])
        q_amax = torch.maximum(q_amax, qg.abs().amax(dim=(1, 2)))
    return k_amax, q_amax


def eval_layer(samples, n_q, n_kv):
    group = n_q // n_kv
    k_amax, q_amax = calibrate(samples, n_q, n_kv)
    head_dim = k_amax.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    base_rl2, sweep = [], {a: [] for a in ALPHAS}
    max_exact_err = 0.0
    for rec in samples:
        q, k, v = rec["q"].float(), rec["k"].float(), rec["v"].float()
        q_last = q[:, -1, :]
        ref = decode_attention(q_last, k, v, scale)
        vq = fake_quant(v, "int8_per_token")          # identical in both arms

        # Baseline: plain symmetric per-token INT8 on K.
        base = decode_attention(q_last, fake_quant(k, "int8_per_token"), vq, scale)
        base_rl2.append(rel_l2(base, ref, dim=-1).mean().item())

        for a in ALPHAS:
            s = smoothquant_factor(k_amax, q_amax, alpha=a)      # [n_kv, hd]
            s_q = s.repeat_interleave(group, dim=0).unsqueeze(1)  # [n_q,1,hd]
            s_k = s.unsqueeze(1)                                  # [n_kv,1,hd]
            qp, kp = apply_smooth(q, k, s_q, s_k)
            # Exactness: smoothed fp output must equal the raw-fp reference.
            sm_fp = decode_attention(qp[:, -1, :], kp, v, scale)
            max_exact_err = max(max_exact_err, rel_l2(sm_fp, ref, dim=-1).max().item())
            out = decode_attention(qp[:, -1, :], fake_quant(kp, "int8_per_token"), vq, scale)
            sweep[a].append(rel_l2(out, ref, dim=-1).mean().item())

    base = sum(base_rl2) / len(base_rl2)
    sweep_m = {a: sum(v) / len(v) for a, v in sweep.items()}
    best_a = min(sweep_m, key=sweep_m.get)
    return {
        "k_outlier_ratio": (k_amax.amax(dim=-1) / k_amax.median(dim=-1).values).mean().item(),
        "baseline_rl2": base,
        "sweep_rl2": sweep_m,
        "best_alpha": best_a,
        "best_rl2": sweep_m[best_a],
        "exactness_max_rl2": max_exact_err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="report/kv_dump.pt")
    ap.add_argument("--out", default="report/smoothquant_eval.json")
    args = ap.parse_args()

    blob = torch.load(args.dump)
    meta, layers = blob["meta"], blob["layers"]
    n_q, n_kv = meta["n_q_heads"], meta["n_kv_heads"]

    report = {"meta": meta, "alphas": ALPHAS, "per_layer": {}}
    cols = " ".join(f"a={a:<4}" for a in ALPHAS)
    print(f"model={meta['model']}  (rel-L2, lower is better)\n")
    print(f"{'layer':>5} {'Kout':>5} {'baseline':>9} | {cols} | {'best':>16}")
    print("-" * (40 + 7 * len(ALPHAS)))

    for idx in sorted(layers):
        r = eval_layer(layers[idx], n_q, n_kv)
        report["per_layer"][idx] = r
        sweep = " ".join(f"{r['sweep_rl2'][a]:.4f}" for a in ALPHAS)
        drop = 100.0 * (1 - r["best_rl2"] / r["baseline_rl2"])
        print(f"{idx:>5} {r['k_outlier_ratio']:>5.0f} {r['baseline_rl2']:>9.4f} | "
              f"{sweep} | a={r['best_alpha']} {r['best_rl2']:.4f} (-{drop:.0f}%)")
        assert r["exactness_max_rl2"] < 1e-3, r["exactness_max_rl2"]

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nexactness check (max smoothed-fp vs ref rel-L2): "
          f"{max(r['exactness_max_rl2'] for r in report['per_layer'].values()):.2e}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
