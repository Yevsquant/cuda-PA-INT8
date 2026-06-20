"""Dump post-RoPE Q/K/V tensors from Qwen2.5-1.5B on real text.

Phase 1 of the quant-accuracy plan: the synthetic N(0,1) baseline lacks the
heavy-tailed, systematic outlier channels that real post-RoPE K exhibits. This
script captures the *real* distribution so fakequant_eval.py can measure how the
symmetric INT8 scheme actually degrades attention output on it.

Capture point: transformers' `eager_attention_forward(module, query, key,
value, ...)`. There `query` is [B, n_q_heads, seq, hd] post-RoPE and `key`/
`value` are [B, n_kv_heads, seq, hd] post-RoPE *before* GQA repeat — exactly the
tensors our kernel consumes. We force `attn_implementation="eager"` so this hook
fires.

Run (on the H200 dev box, sm_90):
    python benchmarks/dump_kv.py --out report/kv_dump.pt
"""

import argparse

import torch
import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
from transformers import AutoModelForCausalLM, AutoTokenizer

# A handful of real, varied natural-language paragraphs. The point is real model
# activations (which produce the systematic outlier channels), not WikiText
# specifically — so we embed text instead of pulling in `datasets`/network.
PROMPTS = [
    "The mitochondrion is a double membrane-bound organelle found in most "
    "eukaryotic cells. It generates most of the cell's supply of adenosine "
    "triphosphate, used as a source of chemical energy. The number of "
    "mitochondria in a cell varies widely by organism, tissue, and cell type.",
    "In 1969, the Apollo 11 mission landed the first humans on the Moon. Neil "
    "Armstrong became the first person to step onto the lunar surface, followed "
    "by Buzz Aldrin. The mission fulfilled a national goal set years earlier and "
    "marked a turning point in the history of space exploration.",
    "Quantization reduces the numerical precision of a model's tensors to shrink "
    "memory and speed up inference. The central difficulty is that activations "
    "contain outliers: a few channels take on values far larger than the rest, "
    "and a naive uniform scale wastes most of the integer range on them.",
    "She walked along the rain-slicked street, collar turned up against the wind, "
    "and thought about everything she had left unsaid. The city hummed around her, "
    "indifferent, its windows glowing like small fires against the early dark.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = "
    "arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = "
    "[x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n"
    "    return quicksort(left) + middle + quicksort(right)",
    "The Treaty of Westphalia, signed in 1648, ended the Thirty Years' War and is "
    "often cited as the origin of the modern system of sovereign states. Its "
    "principles of territorial integrity and non-interference continue to shape "
    "international law and diplomacy to this day.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="report/kv_dump.pt")
    ap.add_argument("--layers", default="0,14,27",
                    help="comma-separated layer indices, or 'all'")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    device = "cuda"
    # bfloat16, NOT fp16: Qwen2.5 has massive activations in deep layers that
    # overflow fp16's 65504 range -> inf -> NaN. bf16 is the model's intended
    # dtype. We capture in bf16 and metric in fp32; the quant study is on the
    # real *values*, independent of the kernel's fp16 storage.
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()

    n_layers = model.config.num_hidden_layers
    want = (set(range(n_layers)) if args.layers == "all"
            else {int(s) for s in args.layers.split(",")})

    captured = {i: [] for i in sorted(want)}
    orig = qwen2_mod.eager_attention_forward

    def hook(module, query, key, value, *a, **kw):
        idx = getattr(module, "layer_idx", None)
        if idx in captured:
            # Drop batch dim; store fp16 on CPU. query: [n_q,seq,hd];
            # key/value: [n_kv,seq,hd] (post-RoPE, pre-GQA-repeat).
            captured[idx].append({
                "q": query[0].detach().to("cpu", torch.bfloat16),
                "k": key[0].detach().to("cpu", torch.bfloat16),
                "v": value[0].detach().to("cpu", torch.bfloat16),
            })
        return orig(module, query, key, value, *a, **kw)

    qwen2_mod.eager_attention_forward = hook
    try:
        with torch.no_grad():
            for text in PROMPTS:
                ids = tok(text, return_tensors="pt",
                          truncation=True, max_length=args.max_tokens).to(device)
                model(**ids)
    finally:
        qwen2_mod.eager_attention_forward = orig

    meta = {
        "model": args.model,
        "n_q_heads": model.config.num_attention_heads,
        "n_kv_heads": model.config.num_key_value_heads,
        "head_dim": model.config.hidden_size // model.config.num_attention_heads,
        "n_prompts": len(PROMPTS),
    }
    torch.save({"meta": meta, "layers": captured}, args.out)

    n = sum(len(v) for v in captured.values())
    print(f"saved {n} (layer,prompt) Q/K/V samples to {args.out}")
    print(f"layers={sorted(captured)} meta={meta}")


if __name__ == "__main__":
    main()
