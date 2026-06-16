"""Offline cross-check (plan §2 / §4 step 5): run ONE cell (ngram, st=4, auto,
ShareGPT) through the offline vLLM `LLM` API and report acceptance + timing, to
confirm the online numbers tie to the same workload. Also demonstrates the §0.5
offline-only fallback shape (one engine in-process; del + empty_cache after).
Online is primary (the go/no-go smoke passed), so this is a sanity check.

  python offline_check.py [--num-prompts 32]
Writes report/specdec/offline_check.json (NOT under cells/, so aggregate.py — which
scans cells/ — ignores it; compare by hand against the matching online cell).
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import configs as C  # noqa: E402
from spec_metrics import _BASES  # noqa: E402


def first_human_prompts(n):
    convs = json.load(open(C.SHAREGPT_PATH))
    out = []
    for c in convs:
        for turn in c.get("conversations", []):
            if turn.get("from") == "human" and turn.get("value"):
                out.append(turn["value"])
                break
        if len(out) >= n:
            break
    return out[:n]


def acceptance_from_metrics(llm):
    """Sum spec-decode counters from LLM.get_metrics() → (rate %, mean accept len)."""
    rev = {v: k for k, v in _BASES.items()}
    tot = {"drafts": 0.0, "draft_tokens": 0.0, "accepted_tokens": 0.0}
    try:
        for mt in llm.get_metrics():
            name = getattr(mt, "name", "")
            name = name[:-6] if name.endswith("_total") else name
            if name in rev:
                tot[rev[name]] += float(getattr(mt, "value", 0) or 0)
    except Exception as e:  # noqa: BLE001 — get_metrics shape is version-sensitive
        print("get_metrics unavailable:", e)
        return None, None
    rate = 100.0 * tot["accepted_tokens"] / tot["draft_tokens"] if tot["draft_tokens"] else None
    alen = 1.0 + tot["accepted_tokens"] / tot["drafts"] if tot["drafts"] else None
    return rate, alen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=32)
    args = ap.parse_args()

    import torch
    from vllm import LLM, SamplingParams

    sc = C.ServerConfig("ngram", "auto", 4)
    prompts = first_human_prompts(args.num_prompts)
    llm = LLM(model=C.MODEL, dtype="bfloat16", max_model_len=C.MAX_MODEL_LEN,
              max_num_seqs=32, speculative_config=sc.speculative_config())
    sp = SamplingParams(temperature=0.0, max_tokens=128)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dur = time.perf_counter() - t0

    ttfts, tpots, n_out = [], [], 0
    for o in outs:
        gen = len(o.outputs[0].token_ids)
        n_out += gen
        m = getattr(o, "metrics", None)
        if m and getattr(m, "first_token_time", None) and getattr(m, "arrival_time", None):
            ttfts.append((m.first_token_time - m.arrival_time) * 1000)
            if getattr(m, "finished_time", None) and gen > 1:
                tpots.append((m.finished_time - m.first_token_time) * 1000 / (gen - 1))

    rate, alen = acceptance_from_metrics(llm)
    rec = {"cell_id": "ngram_st4_auto__sharegpt__offline", "group_id": "ngram_st4_auto",
           "driver": "offline", "method": "ngram", "kv_dtype": "auto", "num_spec_tokens": 4,
           "dataset": "sharegpt", "load": args.num_prompts, "run": 0,
           "num_prompts": len(prompts), "completed": len(outs),
           "output_throughput": n_out / dur if dur else None,
           "median_ttft_ms": statistics.median(ttfts) if ttfts else None,
           "median_tpot_ms": statistics.median(tpots) if tpots else None,
           "acceptance_rate": rate, "acceptance_length": alen,
           "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 1e6}

    out = os.path.join(C._REPO, "report", "specdec", "offline_check.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print("offline:", {k: rec[k] for k in
                       ("median_tpot_ms", "output_throughput", "acceptance_rate", "acceptance_length")})
    print("wrote", out)

    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
