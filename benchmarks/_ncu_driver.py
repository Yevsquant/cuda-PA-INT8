"""Minimal single-config driver for Nsight Compute profiling (Stage 5).

Builds one bandwidth-bound config and launches a chosen variant a few times so
ncu can profile the kernel in isolation (no reference check, no timing loop).

Usage (under ncu):
    ncu --set full -k regex:paged_decode \
        python benchmarks/_ncu_driver.py --variant splitk \
        --batch 32 --ctx 4096 --heads 8 --kv-heads 8
"""

import argparse
import math
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "tests"))

from cuda_ext import VARIANTS  # noqa: E402
from paged_decode_attn import build_paged_kv_cache  # noqa: E402

HEAD_DIM = 128
BLOCK_SIZE = 16
X = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="splitk")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    device = "cuda"
    scale = 1.0 / math.sqrt(HEAD_DIM)
    lens = torch.full((args.batch,), args.ctx, dtype=torch.int32, device=device)
    q = torch.randn(args.batch, args.heads, HEAD_DIM, dtype=torch.float16, device=device)
    k = torch.randn(args.batch, args.ctx, args.kv_heads, HEAD_DIM, dtype=torch.float16, device=device)
    v = torch.randn(args.batch, args.ctx, args.kv_heads, HEAD_DIM, dtype=torch.float16, device=device)
    k_cache, v_cache, block_table = build_paged_kv_cache(k, v, lens, block_size=BLOCK_SIZE, x=X)

    fn = VARIANTS[args.variant]
    out = torch.empty_like(q)
    # warm up (JIT compile + caches), then the reps ncu actually profiles.
    fn(out, q, k_cache, v_cache, block_table, lens, scale, BLOCK_SIZE)
    torch.cuda.synchronize()
    for _ in range(args.reps):
        fn(out, q, k_cache, v_cache, block_table, lens, scale, BLOCK_SIZE)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
