"""Online spec-decode matrix driver (plan §2/§4, online primary per §0.5).

ONE server-config group per invocation: start `vllm serve`, sweep its
load × dataset client cells with `vllm bench serve` (×NUM_RUNS), tear the server
down. Resumable — skips (cell, run) outputs that already exist, so an OOM/crash
costs at most the in-flight group (plan §0.5 Rule 2).

This vLLM (0.19.1) reports spec-decode acceptance directly in the bench-serve
JSON (spec_decode_acceptance_rate / _length), so no /metrics scrape is needed on
the online path; spec_metrics.py remains the scrape-based alternative.

  python run_matrix.py --group ngram_st4_auto   # one group
  python run_matrix.py --resume                 # all groups, skip done cells
  python run_matrix.py --list                   # print groups
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import configs as C  # noqa: E402

OUT = os.path.join(C._REPO, "report", "specdec", "cells")
PORT = 8000
HEALTH_TIMEOUT = 360   # eagle3 may download a head; allow margin


def cell_path(cell, run):
    return os.path.join(OUT, f"{cell.cell_id}__run{run}.json")


def serve_cmd(sc):
    cmd = ["vllm", "serve", C.MODEL, "--dtype", "bfloat16",
           "--max-model-len", str(C.MAX_MODEL_LEN), "--max-num-seqs", "32",
           "--port", str(PORT)]
    if sc.kv_dtype == "fp8":
        cmd += ["--kv-cache-dtype", "fp8"]
    spec = sc.speculative_config()
    if spec is not None:
        cmd += ["--speculative-config", json.dumps(spec)]
    return cmd


def wait_health(proc):
    t0 = time.time()
    while time.time() - t0 < HEALTH_TIMEOUT:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def gpu_mem_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return float(out.decode().splitlines()[0])
    except Exception:
        return None


def run_bench(cell, num_prompts):
    tmp = f"/tmp/bench_{cell.cell_id}.json"
    cmd = ["vllm", "bench", "serve", "--model", C.MODEL,
           "--host", "localhost", "--port", str(PORT),
           *C.dataset_args(cell.dataset),
           "--num-prompts", str(num_prompts),
           "--max-concurrency", str(cell.load),
           "--save-result", "--result-filename", tmp]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    with open(tmp) as f:
        r = json.load(f)
    os.remove(tmp)
    return r


def assemble(cell, run, sc, b):
    return {
        "cell_id": cell.cell_id, "group_id": sc.group_id, "driver": "online",
        "method": sc.method, "kv_dtype": sc.kv_dtype,
        "num_spec_tokens": sc.num_spec_tokens,
        "dataset": cell.dataset, "load": cell.load, "run": run,
        "num_prompts": b.get("num_prompts"), "completed": b.get("completed"),
        "request_throughput": b.get("request_throughput"),
        "output_throughput": b.get("output_throughput"),
        "total_token_throughput": b.get("total_token_throughput"),
        "mean_ttft_ms": b.get("mean_ttft_ms"), "median_ttft_ms": b.get("median_ttft_ms"),
        "p99_ttft_ms": b.get("p99_ttft_ms"),
        "mean_tpot_ms": b.get("mean_tpot_ms"), "median_tpot_ms": b.get("median_tpot_ms"),
        # spec_decode_* absent for baseline -> None
        "acceptance_rate": b.get("spec_decode_acceptance_rate"),
        "acceptance_length": b.get("spec_decode_acceptance_length"),
        "peak_gpu_mem_mb": gpu_mem_mb(),
    }


def teardown(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=40)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def run_group(group_id, num_prompts, num_runs):
    os.makedirs(OUT, exist_ok=True)
    sc = next((s for s in C.server_configs() if s.group_id == group_id), None)
    if sc is None:
        raise KeyError(group_id)
    cells = C.cells_for_group(group_id)
    todo = [(c, r) for c in cells for r in range(num_runs)
            if not os.path.exists(cell_path(c, r))]
    if not todo:
        print(f"[{group_id}] all {len(cells) * num_runs} runs present; skip")
        return
    print(f"[{group_id}] server up for {len(todo)} pending runs: {' '.join(serve_cmd(sc))}")
    log = open(f"/tmp/serve_{group_id}.log", "w")
    proc = subprocess.Popen(serve_cmd(sc), stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        if not wait_health(proc):
            print(f"[{group_id}] SERVER FAILED TO START (see /tmp/serve_{group_id}.log) — "
                  f"skipping group (eagle3 head may be unavailable, Decision 1)")
            return
        print(f"[{group_id}] SERVER_READY")
        for c, r in todo:
            try:
                b = run_bench(c, num_prompts)
            except subprocess.CalledProcessError as e:
                print(f"  {c.cell_id} run{r}: bench FAILED ({e}) — skip")
                continue
            rec = assemble(c, r, sc, b)
            with open(cell_path(c, r), "w") as f:
                json.dump(rec, f, indent=2)
            print(f"  {c.cell_id} run{r}: tpot={rec['median_tpot_ms']:.2f}ms "
                  f"thr={rec['output_throughput']:.1f}tok/s acc={rec['acceptance_rate']}")
    finally:
        teardown(proc)
        print(f"[{group_id}] server down")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--num-prompts", type=int, default=C.NUM_PROMPTS,
                    help="override prompts/cell (default %(default)s; lower for a quick run)")
    ap.add_argument("--runs", type=int, default=C.NUM_RUNS,
                    help="override runs/cell (default %(default)s)")
    args = ap.parse_args()
    if args.list:
        for g in C.group_ids():
            print(g)
        return
    if args.group:
        run_group(args.group, args.num_prompts, args.runs)
    elif args.resume:
        for g in C.group_ids():
            run_group(g, args.num_prompts, args.runs)
    else:
        ap.error("pass --group <id>, --resume, or --list")


if __name__ == "__main__":
    main()
