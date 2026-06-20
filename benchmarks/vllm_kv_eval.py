"""vLLM native KV-cache-dtype GSM8K, to validate the HF fake-quant FP8 finding
(Phase 6).

Our HF harness found FP8-E4M3 KV badly hurts Qwen2.5-1.5B (E4M3's ~6% per-element
relative error lands on the large K components the softmax is most sensitive to,
where INT8's max-anchored scale stays <1%). That contradicts the folk wisdom that
FP8 KV "just works", so we check it against vLLM's *real* FP8 KV kernel.

vLLM has no INT8 KV option (the gap this project targets) — its only sub-fp16 KV
dtype is fp8. So this measures vLLM(auto/bf16) vs vLLM(fp8); the INT8 numbers come
from the HF harness (ppl_eval / gsm8k_eval).

Run once per dtype (separate processes keep GPU memory clean on a shared card):
    python benchmarks/vllm_kv_eval.py --kv-dtype auto --n 100
    python benchmarks/vllm_kv_eval.py --kv-dtype fp8  --n 100
"""

import argparse
import json
import os
import sys

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(__file__))
from gsm8k_eval import INSTRUCT, _num, extract_pred  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--kv-dtype", default="auto", choices=["auto", "fp8"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--gpu-mem", type=float, default=0.12)  # shared GPU: keep low
    ap.add_argument("--out", default="report/vllm_kv_eval.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": q + INSTRUCT}],
                                tokenize=False, add_generation_prompt=True)
        for q in ds["question"]
    ]
    golds = [_num(a.split("####")[-1]) for a in ds["answer"]]

    llm = LLM(model=args.model, dtype="bfloat16", kv_cache_dtype=args.kv_dtype,
              gpu_memory_utilization=args.gpu_mem, max_model_len=2048,
              enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=320, repetition_penalty=1.0)
    outs = llm.generate(prompts, sp)

    correct = 0
    for o, gold in zip(outs, golds):
        pred = extract_pred(o.outputs[0].text)
        if pred is not None and gold is not None and abs(pred - gold) < 1e-4:
            correct += 1
    acc = correct / args.n

    rec = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            rec = json.load(f)
    rec[args.kv_dtype] = {"acc": acc, "n": args.n, "engine": "vllm"}
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"\nvLLM kv_cache_dtype={args.kv_dtype}: GSM8K acc={acc:.3f} (n={args.n})")


if __name__ == "__main__":
    main()
