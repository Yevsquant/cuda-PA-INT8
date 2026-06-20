"""Compare outlier transforms on real K/V: baseline vs SmoothQuant vs Hadamard
vs SmoothQuant+Hadamard (Phase 3).

All arms quantize V identically (per-token) and use the same raw-fp reference, so
the only difference is the K path. Each transform is exact on the dot product
(checked), so it changes K's quantizability, not the scores — no kernel change.
SmoothQuant migrates per-channel difficulty to fp Q; Hadamard spreads per-token
outlier energy across channels; the composition does SmoothQuant first (in the
outlier-aligned channel basis) then rotates.

    python benchmarks/transform_eval.py --dump report/kv_dump.pt --alpha 0.85
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from quant_metrics import rel_l2  # noqa: E402
from quant_transforms import (  # noqa: E402
    apply_smooth, hadamard_apply, hadamard_matrix, smoothquant_factor,
)
from fakequant_eval import decode_attention, fake_quant  # noqa: E402
from smoothquant_eval import calibrate  # noqa: E402

ARMS = ["baseline", "smoothquant", "hadamard", "smooth+had"]


def eval_layer(samples, n_q, n_kv, alpha, H):
    group = n_q // n_kv
    k_amax, q_amax = calibrate(samples, n_q, n_kv)
    s = smoothquant_factor(k_amax, q_amax, alpha=alpha)        # [n_kv, hd]
    s_q = s.repeat_interleave(group, dim=0).unsqueeze(1)       # [n_q,1,hd]
    s_k = s.unsqueeze(1)                                       # [n_kv,1,hd]
    scale = 1.0 / math.sqrt(H.shape[0])

    acc = {a: 0.0 for a in ARMS}
    exact = 0.0
    for rec in samples:
        q, k, v = rec["q"].float(), rec["k"].float(), rec["v"].float()
        ref = decode_attention(q[:, -1, :], k, v, scale)
        vq = fake_quant(v, "int8_per_token")

        # Build each arm's (q', k') in fp, then quantize only k'.
        qs, ks = apply_smooth(q, k, s_q, s_k)
        qh, kh = hadamard_apply(q, k, H)
        qsh, ksh = hadamard_apply(qs, ks, H)
        arms = {
            "baseline":    (q, k),
            "smoothquant": (qs, ks),
            "hadamard":    (qh, kh),
            "smooth+had":  (qsh, ksh),
        }
        for name, (qa, ka) in arms.items():
            # Exactness: transformed-fp output must match the raw-fp reference.
            fp_out = decode_attention(qa[:, -1, :], ka, v, scale)
            exact = max(exact, rel_l2(fp_out, ref, dim=-1).max().item())
            out = decode_attention(qa[:, -1, :], fake_quant(ka, "int8_per_token"), vq, scale)
            acc[name] += rel_l2(out, ref, dim=-1).mean().item() / len(samples)

    return acc, exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="report/kv_dump.pt")
    ap.add_argument("--out", default="report/transform_eval.json")
    ap.add_argument("--alpha", type=float, default=0.85)
    args = ap.parse_args()

    blob = torch.load(args.dump)
    meta, layers = blob["meta"], blob["layers"]
    n_q, n_kv, hd = meta["n_q_heads"], meta["n_kv_heads"], meta["head_dim"]
    H = hadamard_matrix(hd)

    report = {"meta": meta, "alpha": args.alpha, "per_layer": {}}
    print(f"model={meta['model']}  SmoothQuant alpha={args.alpha}  "
          f"(decode rel-L2, lower is better)\n")
    print(f"{'layer':>5} " + " ".join(f"{a:>12}" for a in ARMS))
    print("-" * (6 + 13 * len(ARMS)))

    max_exact = 0.0
    for idx in sorted(layers):
        acc, exact = eval_layer(layers[idx], n_q, n_kv, args.alpha, H)
        max_exact = max(max_exact, exact)
        report["per_layer"][idx] = acc
        base = acc["baseline"]
        cells = []
        for a in ARMS:
            change = 100.0 * (acc[a] / base - 1.0)  # negative == error reduced
            cells.append(f"{acc[a]:.4f}" + ("" if a == "baseline" else f"({change:+.0f}%)"))
        print(f"{idx:>5} " + " ".join(f"{c:>12}" for c in cells))

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nexactness check (max transformed-fp vs ref rel-L2): {max_exact:.2e}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
