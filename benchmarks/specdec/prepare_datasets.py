"""Build the spec-decode benchmark datasets on disk (plan §3), low-RAM.

ShareGPT: download the standard cleaned-split json, then **stream-parse** (ijson)
a fixed-seed reservoir sample of ~200 conversations to data/sharegpt_subset.json,
and delete the raw download. Never `json.load` the ~650 MB file — on the 8 GB
host that alone risks OOM (plan §0.5).
HumanEval: fetch the canonical 164 problems (gzipped jsonl, stdlib only — the env
has no `datasets`/`pyarrow`) and write prompts to data/humaneval.jsonl.

Commit the manifest (data/sharegpt_manifest.json), not the raw json.
Run:  python prepare_datasets.py   (one-time network fetch; verifies counts)
"""

import gzip
import json
import os
import random
import urllib.request

import ijson
from huggingface_hub import hf_hub_download

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
DATA = os.path.join(_REPO, "data")

SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
SUBSET_N = 200
SEED = 0
# openai/human-eval ships the canonical 164-problem set as a gzipped jsonl.
HUMANEVAL_URLS = [
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
    "https://raw.githubusercontent.com/openai/human-eval/main/data/HumanEval.jsonl.gz",
]


def prepare_sharegpt() -> int:
    os.makedirs(DATA, exist_ok=True)
    raw = hf_hub_download(repo_id=SHAREGPT_REPO, filename=SHAREGPT_FILE,
                          repo_type="dataset", local_dir=DATA)
    rng = random.Random(SEED)
    reservoir = []   # reservoir sample (algorithm R): only SUBSET_N items in RAM
    n_seen = 0
    with open(raw, "rb") as f:
        for conv in ijson.items(f, "item"):
            if not conv.get("conversations"):
                continue
            if len(reservoir) < SUBSET_N:
                reservoir.append((n_seen, conv))
            else:
                j = rng.randint(0, n_seen)   # inclusive
                if j < SUBSET_N:
                    reservoir[j] = (n_seen, conv)
            n_seen += 1

    subset = [c for _, c in reservoir]
    out = os.path.join(DATA, "sharegpt_subset.json")
    with open(out, "w") as f:
        json.dump(subset, f)
    with open(os.path.join(DATA, "sharegpt_manifest.json"), "w") as f:
        json.dump({"source": f"{SHAREGPT_REPO}/{SHAREGPT_FILE}", "seed": SEED,
                   "n_seen": n_seen, "n_subset": len(subset),
                   "ids": [i for i, _ in reservoir]}, f, indent=2)
    os.remove(raw)   # delete raw download (plan §3)
    print(f"sharegpt: sampled {len(subset)} of {n_seen} convs -> {out} (raw removed)")
    return len(subset)


def prepare_humaneval() -> int:
    os.makedirs(DATA, exist_ok=True)
    last_err = None
    for url in HUMANEVAL_URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                blob = r.read()
            break
        except Exception as e:   # noqa: BLE001 — try the next mirror
            last_err = e
    else:
        raise RuntimeError(f"could not fetch HumanEval from any mirror: {last_err}")

    probs = [json.loads(l) for l in gzip.decompress(blob).decode().splitlines() if l.strip()]
    out = os.path.join(DATA, "humaneval.jsonl")
    with open(out, "w") as f:
        for p in probs:
            f.write(json.dumps({"prompt": p["prompt"]}) + "\n")
    print(f"humaneval: wrote {len(probs)} prompts -> {out}")
    return len(probs)


def main():
    n_sg = prepare_sharegpt()
    n_he = prepare_humaneval()
    assert 150 <= n_sg <= 250, f"sharegpt subset {n_sg} outside expected ~200"
    assert n_he == 164, f"humaneval count {n_he} != 164"
    assert not os.path.exists(os.path.join(DATA, SHAREGPT_FILE)), "raw not removed"
    print("OK datasets ready")


if __name__ == "__main__":
    main()
