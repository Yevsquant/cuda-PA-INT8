"""End-to-end GSM8K accuracy with fake-quant KV (Phase 5, generation arm).

Complements the perplexity table with a downstream task: does the INT8 KV
degradation actually cost correct answers, and does SmoothQuant recover them?
Same eager_attention_forward fake-quant hook as ppl_eval, now under batched
greedy generation.

repetition_penalty is forced to 1.0: Qwen2.5's generation_config defaults it to
1.1 even under do_sample=False, which would make this not-quite-greedy and muddy
a clean cross-mode comparison.

    python benchmarks/gsm8k_eval.py --n 200
"""

import argparse
import json
import os
import re
import sys

import torch
import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from quant_transforms import hadamard_matrix  # noqa: E402
from ppl_eval import make_hook  # noqa: E402

MODES = ["baseline", "int8_per_token", "int8_smoothquant", "int8_asym"]
INSTRUCT = ("\nPlease reason step by step, and put your final numeric answer "
            "after '####'.")


def _num(s):
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def extract_pred(text):
    """Final answer: the number after the last '####', else the last number."""
    if "####" in text:
        tail = text.split("####")[-1]
        m = re.search(r"-?\d[\d,]*\.?\d*", tail)
        if m:
            return _num(m.group())
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return _num(nums[-1]) if nums else None


def run_mode(model, tok, prompts, golds, mode, alpha, H, orig, batch, max_new):
    qwen2_mod.eager_attention_forward = (
        orig if mode == "baseline" else make_hook(orig, mode, alpha, H))
    correct = 0
    try:
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=max_new, do_sample=False,
                    repetition_penalty=1.0, pad_token_id=tok.eos_token_id)
            gen = tok.batch_decode(out[:, enc.input_ids.shape[1]:],
                                   skip_special_tokens=True)
            for g, gold in zip(gen, golds[i:i + batch]):
                pred = extract_pred(g)
                if pred is not None and abs(pred - gold) < 1e-4:
                    correct += 1
    finally:
        qwen2_mod.eager_attention_forward = orig
    return correct / len(prompts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="report/gsm8k_eval.json")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=320)
    args = ap.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"  # decoder-only batched generation needs left pad
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": q + INSTRUCT}],
            tokenize=False, add_generation_prompt=True)
        for q in ds["question"]
    ]
    golds = [_num(a.split("####")[-1]) for a in ds["answer"]]

    H = hadamard_matrix(model.config.hidden_size // model.config.num_attention_heads,
                        device=device)
    orig = qwen2_mod.eager_attention_forward

    report = {"model": args.model, "alpha": args.alpha, "n": args.n, "acc": {}}
    print(f"model={args.model}  GSM8K n={args.n}  (higher acc is better)\n")
    print(f"{'mode':>18} {'acc':>7} {'dAcc':>7}")
    print("-" * 35)

    base = None
    for mode in MODES:
        acc = run_mode(model, tok, prompts, golds, mode, args.alpha, H, orig,
                       args.batch, args.max_new)
        report["acc"][mode] = acc
        if base is None:
            base = acc
        print(f"{mode:>18} {acc:>7.3f} {acc - base:>+7.3f}")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
