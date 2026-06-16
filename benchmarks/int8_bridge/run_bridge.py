"""INT8 end-to-end bridge — ONE cache variant per process (plan §7, 8 GB rule).

Runs real Qwen2.5-1.5B-Instruct greedy decode through the custom paged-decode
kernels for a single cache variant and dumps its metrics to
`report/int8_bridge_<variant>.json`. `combine_bridge.py` merges the three
per-variant JSONs into the comparison report. Never holds two models/caches in
host RAM at once (the active variant only).

The fp16 run is the **reference**: its greedy continuation is what the int8
variants are scored against (greedy-match + perplexity), so each variant
teacher-forces the fp16 tokens. Therefore run `--variant fp16` FIRST.

Variants share the identical prefill (bf16 SDPA) + greedy-decode loop; only the
decode-step KV dtype and kernel differ:
    fp16             — FP16 split-K kernel (non-quantized baseline / reference)
    int8_per_token   — per-(token,kv_head) scales
    int8_per_tensor  — one static layer-global scale (ablation)

Run:  python run_bridge.py --variant fp16            # reference, full
      python run_bridge.py --variant int8_per_token  # needs the fp16 json
      python run_bridge.py --variant int8_per_tensor
      add --quick for a 2-prompt / 16-token smoke (use the SAME flag for all 3)
"""

import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
import qwen_paged_attn as bridge  # noqa: E402
from paged_cache import PagedKVCache  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
VARIANTS = ["fp16", "int8_per_token", "int8_per_tensor"]
REF_LEN = 2048  # geometry-clean headline-memory length

# A handful of representative prompts: chat (ShareGPT-like) + code (HumanEval-like).
PROMPTS = [
    ("chat", "Explain what PagedAttention is in two sentences."),
    ("chat", "What are three benefits of quantizing the KV cache to INT8?"),
    ("chat", "Summarize the plot of Romeo and Juliet in one paragraph."),
    ("code", "Write a Python function `is_prime(n)` that returns True iff n is prime."),
    ("code", "Write a Python function `fib(n)` returning the n-th Fibonacci number."),
]


def build_cache(cfg, variant, max_len):
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    return PagedKVCache(
        num_layers=cfg.num_hidden_layers, num_kv_heads=cfg.num_key_value_heads,
        head_dim=head_dim, max_len=max_len, variant=variant)


def encode(tok, text):
    msgs = [{"role": "user", "content": text}]
    out = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids = out if torch.is_tensor(out) else out["input_ids"]  # BatchEncoding in tf 5.x
    return ids.to("cuda")


def fp16_ref_path():
    return os.path.join(_REPO, "report", "int8_bridge_fp16.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    variant = args.variant
    prompts = PROMPTS[:2] if args.quick else PROMPTS
    max_new = 16 if args.quick else args.max_new

    # int8 variants are scored against the fp16 reference continuation.
    ref_cells = None
    if variant != "fp16":
        if not os.path.exists(fp16_ref_path()):
            sys.exit(f"need {fp16_ref_path()} — run `--variant fp16"
                     f"{' --quick' if args.quick else ''}` first")
        with open(fp16_ref_path()) as f:
            ref = json.load(f)
        ref_cells = ref["cells"]
        if len(ref_cells) != len(prompts) or ref["max_new"] != max_new:
            sys.exit("fp16 reference shape mismatch — re-run fp16 with the same "
                     "--quick / --max-new")

    tok = AutoTokenizer.from_pretrained(MODEL)
    bridge.register()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation=bridge.ATTN_NAME).cuda().eval()
    cfg = model.config

    cells = []
    for i, (tag, text) in enumerate(prompts):
        ids = encode(tok, text)
        plen = ids.shape[1]
        cache = build_cache(cfg, variant, plen + max_new)

        gen, ttft, tpot, _ = bridge.generate_greedy(model, ids, max_new, cache)
        gen_tokens = gen[0].tolist()
        kv_bytes = cache.kv_bytes_for_len(plen + max_new)

        # Score against the fp16 continuation (its own, when this IS fp16).
        if variant == "fp16":
            ref_tokens = gen_tokens
        else:
            if ref_cells[i]["prompt"] != text:
                sys.exit("prompt order mismatch vs fp16 reference")
            ref_tokens = ref_cells[i]["gen_tokens"]
        ref_t = torch.tensor([ref_tokens], device=ids.device)
        # ref length == max_new == cache capacity beyond the prompt, so reuse cache.
        logps = bridge.teacher_force_logprobs(model, ids, ref_t, cache)

        cells.append(dict(tag=tag, prompt=text, prompt_len=plen,
                          gen_tokens=gen_tokens, ttft=ttft, tpot=tpot,
                          kv_bytes=kv_bytes, fp16_logprobs=logps))
        ppl = float(torch.exp(torch.tensor(-sum(logps) / len(logps))))
        print(f"[{variant}|{tag}] plen={plen} tpot={1e3 * tpot:.2f}ms ppl={ppl:.3f}")

    kv_bytes_ref = build_cache(cfg, variant, REF_LEN).kv_bytes_for_len(REF_LEN)
    out = dict(variant=variant, model=MODEL, max_new=max_new,
               mem_ref_len=REF_LEN, kv_bytes_ref=kv_bytes_ref, cells=cells)
    path = os.path.join(_REPO, "report", f"int8_bridge_{variant}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
