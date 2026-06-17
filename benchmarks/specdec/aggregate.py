"""Combine step (plan §2/§4): stream per-(cell,run) JSONs one at a time into a
tidy results.csv (median across runs) + a summary markdown. Never glob-loads all
records into memory beyond the small per-cell metric dicts (plan §0.5 Rule 2).

  python aggregate.py
Writes report/specdec_results.csv and report/specdec_benchmark.md.
"""

import csv
import json
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import configs as C  # noqa: E402

CELLS = os.path.join(C._REPO, "report", "specdec", "cells")
CSV_OUT = os.path.join(C._REPO, "report", "specdec_results.csv")
MD_OUT = os.path.join(C._REPO, "report", "specdec_benchmark.md")

# (key, label) metrics medianed across runs.
METRICS = [
    ("median_ttft_ms", "TTFT_ms"),
    ("median_tpot_ms", "TPOT_ms"),
    ("output_throughput", "out_tok_s"),
    ("total_token_throughput", "tot_tok_s"),
    ("acceptance_rate", "accept_%"),
    ("acceptance_length", "accept_len"),
    ("peak_gpu_mem_mb", "gpu_mem_MB"),
]


def _median(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return statistics.median(vals) if vals else None


def load_cells():
    """Stream per-run JSONs, grouping metric values by cell_id. Holds only small
    lists of scalars, never the raw files."""
    if not os.path.isdir(CELLS):
        raise SystemExit(f"no cells dir {CELLS} — run run_matrix.py first")
    by_cell = {}
    for name in sorted(os.listdir(CELLS)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(CELLS, name)) as f:
            rec = json.load(f)
        cid = rec["cell_id"]
        slot = by_cell.setdefault(cid, {"meta": rec, "runs": 0,
                                        **{k: [] for k, _ in METRICS}})
        slot["runs"] += 1
        for k, _ in METRICS:
            slot[k].append(rec.get(k))
    return by_cell


def main():
    by_cell = load_cells()
    if not by_cell:
        raise SystemExit("no cell JSONs found")

    rows = []
    for cid, slot in by_cell.items():
        m = slot["meta"]
        row = {"cell_id": cid, "group_id": m["group_id"], "method": m["method"],
               "kv_dtype": m["kv_dtype"], "num_spec_tokens": m["num_spec_tokens"],
               "dataset": m["dataset"], "load": m["load"], "n_runs": slot["runs"]}
        for k, label in METRICS:
            row[label] = _median(slot[k])
        rows.append(row)
    rows.sort(key=lambda r: (r["dataset"], r["load"], r["group_id"]))

    cols = (["cell_id", "group_id", "method", "kv_dtype", "num_spec_tokens",
             "dataset", "load", "n_runs"] + [label for _, label in METRICS])
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {CSV_OUT} ({len(rows)} cells, "
          f"{len(C.all_cells())} in full matrix)")

    write_report(rows)


def _fmt(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def write_report(rows):
    lines = ["# Speculative-decoding benchmark — Qwen2.5-1.5B-Instruct (vLLM, A100-80GB)\n"]
    lines.append("Online `vllm bench serve` driver; median of up to "
                 f"{C.NUM_RUNS} runs/cell. Acceptance from the bench-serve JSON "
                 "(`spec_decode_acceptance_rate/_length`). Cells grouped by "
                 "server-config, one server lifetime each (plan §0.5).\n")
    n_done = len(rows)
    n_full = len(C.all_cells())
    present_groups = {r["group_id"] for r in rows}
    missing = [g for g in C.group_ids() if g not in present_groups]
    if missing:
        non_eagle = [g for g in missing if not g.startswith("eagle3")]
        if not non_eagle:
            lines.append(
                f"> **{n_done}/{n_full} cells — complete deliverable.** The "
                f"{len(missing)} EAGLE-3 group(s) ({', '.join(missing)}) are excluded: "
                "no valid Qwen2.5-1.5B EAGLE-3 head exists (Decision 1); n-gram is the "
                "primary method.\n")
        else:
            lines.append(
                f"> **Partial:** {n_done}/{n_full} cells; missing groups: "
                f"{', '.join(missing)}. Re-run `run_matrix.py --resume`, then re-aggregate.\n")

    hdr = "| group | method | kv | spec_tok | acc % | acc_len | TTFT ms | TPOT ms | out tok/s | GPU MB |"
    sep = "|" + "---|" * 10
    for ds in C.DATASETS:
        for load in C.LOADS:
            cells = [r for r in rows if r["dataset"] == ds and r["load"] == load]
            if not cells:
                continue
            lines.append(f"\n## {ds} — load (max-concurrency) {load}\n")
            lines.append(hdr)
            lines.append(sep)
            for r in cells:
                lines.append(
                    f"| {r['group_id']} | {r['method']} | {r['kv_dtype']} | "
                    f"{r['num_spec_tokens'] if r['num_spec_tokens'] is not None else '—'} | "
                    f"{_fmt(r['accept_%'],1)} | {_fmt(r['accept_len'])} | "
                    f"{_fmt(r['TTFT_ms'],1)} | {_fmt(r['TPOT_ms'])} | "
                    f"{_fmt(r['out_tok_s'],1)} | {_fmt(r['gpu_mem_MB'],0)} |")
    lines.append("\n## Caveats\n")
    lines.append("- n-gram is the primary spec method; EAGLE-3 is best-effort with a "
                 "community 1.5B head (Decision 1) — absent rows mean its server failed to start.")
    lines.append("- 8 GB host: online server + bench client coexist (go/no-go smoke passed); "
                 "runs are strictly sequential, one server at a time.")
    lines.append("- GPU MB is the server's steady allocation sampled per cell (nvidia-smi), not a true per-request peak.")
    with open(MD_OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {MD_OUT}")


if __name__ == "__main__":
    main()
