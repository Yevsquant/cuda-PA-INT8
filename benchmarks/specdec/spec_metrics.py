"""Scrape vLLM's Prometheus /metrics for spec-decode acceptance (plan §2).

`vllm bench serve` JSON has no acceptance rate; the counters live in /metrics:
  vllm:spec_decode_num_drafts(_total)          — draft steps (proposals)
  vllm:spec_decode_num_draft_tokens(_total)    — tokens proposed
  vllm:spec_decode_num_accepted_tokens(_total) — draft tokens accepted
Diff the counters before/after a run (they are monotonic Prometheus counters):
  acceptance_rate = Δaccepted / Δdraft_tokens
  mean_accept_len = 1 + Δaccepted / Δdrafts    (avg accepted per step + bonus token)

Used by run_matrix.py (online) around each cell; offline_check.py gets the same
numbers from LLM.get_metrics() instead.
"""

import urllib.request

_BASES = {
    "drafts": "vllm:spec_decode_num_drafts",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens",
}


def fetch_counters(port: int, host: str = "localhost", timeout: float = 5.0) -> dict:
    """Sum each spec-decode counter across all label sets. Absent counter → 0.0
    (baseline runs emit none)."""
    url = f"http://{host}:{port}/metrics"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        text = r.read().decode()
    return parse_counters(text)


def parse_counters(text: str) -> dict:
    totals = {k: 0.0 for k in _BASES}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        metric = name.split("{", 1)[0]
        if metric.endswith("_total"):
            metric = metric[: -len("_total")]
        for key, base in _BASES.items():
            if metric == base:
                try:
                    totals[key] += float(value)
                except ValueError:
                    pass
    return totals


def acceptance(before: dict, after: dict) -> dict:
    d_drafts = after["drafts"] - before["drafts"]
    d_dtok = after["draft_tokens"] - before["draft_tokens"]
    d_acc = after["accepted_tokens"] - before["accepted_tokens"]
    return {
        "delta_drafts": d_drafts,
        "delta_draft_tokens": d_dtok,
        "delta_accepted_tokens": d_acc,
        "acceptance_rate": (d_acc / d_dtok) if d_dtok else None,
        "mean_accept_len": (1.0 + d_acc / d_drafts) if d_drafts else None,
    }


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    print(json.dumps(fetch_counters(ap.parse_args().port), indent=2))
