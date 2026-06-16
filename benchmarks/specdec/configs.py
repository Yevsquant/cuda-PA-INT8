"""The spec-decode benchmark matrix — single source of truth (plan §1).

Cells are grouped by *server-config* (method × kv_dtype × num_spec_tokens): one
`vllm serve` lifetime per group, sweeping its load × dataset client cells, then
torn down (plan §0.5 Rule 1). run_matrix.py consumes this; aggregate.py keys on
`cell_id`.

Pruning (plan §1, ≈60 cells):
  - baseline: no num_speculative_tokens axis → {auto, fp8} = 2 groups.
  - ngram (primary/strongest): sweep num_spec_tokens {2,4,8} × kv {auto, fp8} = 6 groups.
  - eagle3 (best-effort, weaker — Decision 1): fix num_spec_tokens=4 × kv {auto, fp8} = 2 groups.
  Client cells per group: load {1,8,32} × dataset {sharegpt, humaneval} = 6.
  → 10 groups × 6 = 60 cells, ×NUM_RUNS (median).
"""

import os
from dataclasses import dataclass

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_MODEL_LEN = 4096
NUM_PROMPTS = 256        # per-cell client cap (plan §0.5: ≤256)
NUM_RUNS = 3             # median of 3
LOADS = (1, 8, 32)       # max_concurrency: batch=1 latency; 8/32 throughput
DATASETS = ("sharegpt", "humaneval")

# Decision 1: no official Qwen2.5-1.5B EAGLE-3 head exists; community head, best-effort.
EAGLE3_HEAD = "JerryGJX/TLT-Eagle3-Qwen2.5-1.5B-76000"

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_REPO, "data")
SHAREGPT_PATH = os.path.join(DATA_DIR, "sharegpt_subset.json")
HUMANEVAL_PATH = os.path.join(DATA_DIR, "humaneval.jsonl")


@dataclass(frozen=True)
class ServerConfig:
    method: str                          # baseline | ngram | eagle3
    kv_dtype: str                        # auto | fp8
    num_spec_tokens: int | None = None   # None for baseline

    @property
    def group_id(self) -> str:
        st = "" if self.num_spec_tokens is None else f"_st{self.num_spec_tokens}"
        return f"{self.method}{st}_{self.kv_dtype}"

    def speculative_config(self) -> dict | None:
        """The `--speculative-config` JSON for `vllm serve` (None for baseline)."""
        if self.method == "baseline":
            return None
        if self.method == "ngram":
            return {"method": "ngram", "num_speculative_tokens": self.num_spec_tokens,
                    "prompt_lookup_max": 3, "prompt_lookup_min": 1}
        if self.method == "eagle3":
            return {"method": "eagle3", "model": EAGLE3_HEAD,
                    "num_speculative_tokens": self.num_spec_tokens}
        raise ValueError(f"unknown method {self.method!r}")


@dataclass(frozen=True)
class Cell:
    server: ServerConfig
    load: int                            # max_concurrency
    dataset: str                         # sharegpt | humaneval

    @property
    def cell_id(self) -> str:
        return f"{self.server.group_id}__{self.dataset}__b{self.load}"


def server_configs() -> list[ServerConfig]:
    out = [ServerConfig("baseline", kv) for kv in ("auto", "fp8")]
    out += [ServerConfig("ngram", kv, st) for kv in ("auto", "fp8") for st in (2, 4, 8)]
    out += [ServerConfig("eagle3", kv, 4) for kv in ("auto", "fp8")]
    return out


def group_ids() -> list[str]:
    return [s.group_id for s in server_configs()]


def cells_for_group(group_id: str) -> list[Cell]:
    sc = next((s for s in server_configs() if s.group_id == group_id), None)
    if sc is None:
        raise KeyError(group_id)
    return [Cell(sc, load, ds) for ds in DATASETS for load in LOADS]


def all_cells() -> list[Cell]:
    return [c for g in group_ids() for c in cells_for_group(g)]


def dataset_args(dataset: str) -> list[str]:
    """`vllm bench serve` dataset selection for a cell's dataset."""
    if dataset == "sharegpt":
        return ["--dataset-name", "sharegpt", "--dataset-path", SHAREGPT_PATH]
    if dataset == "humaneval":
        return ["--dataset-name", "custom", "--dataset-path", HUMANEVAL_PATH]
    raise ValueError(f"unknown dataset {dataset!r}")


if __name__ == "__main__":
    gs, cs = group_ids(), all_cells()
    print(f"{len(gs)} server groups, {len(cs)} cells, ×{NUM_RUNS} runs")
    for g in gs:
        print(f"  {g:22s} -> {len(cells_for_group(g))} cells")
