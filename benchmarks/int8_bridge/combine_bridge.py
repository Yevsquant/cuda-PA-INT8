"""Merge the three per-variant int8-bridge JSONs into the comparison report.

fp16 is the reference: greedy-match and perplexity are scored against its greedy
continuation (each variant already teacher-forced the fp16 tokens in run_bridge,
dumping per-step logprobs). This step holds no model — just the small JSONs.

Run (after the three `run_bridge.py --variant ...` invocations):
    python combine_bridge.py
Writes report/int8_bridge.md + report/int8_bridge_results.json.
"""

import json
import math
import os
import statistics

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
VARIANTS = ["fp16", "int8_per_token", "int8_per_tensor"]


def load(variant):
    path = os.path.join(_REPO, "report", f"int8_bridge_{variant}.json")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} — run: python run_bridge.py --variant {variant}")
    with open(path) as f:
        return json.load(f)


def cell_ppl(cell):
    lp = cell["fp16_logprobs"]
    return math.exp(-sum(lp) / len(lp))


def greedy_match(gen, ref):
    m = min(len(gen), len(ref))
    return sum(int(a == b) for a, b in zip(gen[:m], ref[:m])) / m


def main():
    data = {v: load(v) for v in VARIANTS}
    n = len(data["fp16"]["cells"])
    for v in VARIANTS:
        if len(data[v]["cells"]) != n:
            raise SystemExit(
                f"{v} has {len(data[v]['cells'])} cells but fp16 has {n} — "
                "re-run the variants with matching --quick / --max-new")
    ref_cells = data["fp16"]["cells"]
    fp16_kv = data["fp16"]["kv_bytes_ref"]

    summary = {}
    for v in VARIANTS:
        cells = data[v]["cells"]
        matches = [greedy_match(cells[i]["gen_tokens"], ref_cells[i]["gen_tokens"])
                   for i in range(n)]
        kv_ref = data[v]["kv_bytes_ref"]
        summary[v] = dict(
            ttft_ms=1e3 * statistics.median(c["ttft"] for c in cells),
            tpot_ms=1e3 * statistics.median(c["tpot"] for c in cells),
            greedy_match=statistics.mean(matches),
            ppl=statistics.median(cell_ppl(c) for c in cells),
            kv_bytes_ref=kv_ref,
            kv_ratio_vs_fp16=kv_ref / fp16_kv,
        )

    ref_len = data["fp16"]["mem_ref_len"]
    max_new = data["fp16"]["max_new"]
    out_json = os.path.join(_REPO, "report", "int8_bridge_results.json")
    with open(out_json, "w") as f:
        json.dump(dict(model=MODEL, max_new=max_new, num_prompts=n,
                       mem_ref_len=ref_len, summary=summary), f, indent=2)
    print("wrote", out_json)

    write_report(summary, ref_len, n, max_new)


def write_report(summary, ref_len, n, max_new):
    def mb(b):
        return b / (1024 ** 2)

    lines = []
    lines.append("# INT8 KV-cache end-to-end bridge — Qwen2.5-1.5B-Instruct\n")
    lines.append(
        "One honest end-to-end data point for the custom per-token INT8 KV-cache kernel "
        "on real Qwen weights. A standalone harness (not vLLM) routes Qwen2's **decode-step** "
        "attention through the custom split-K paged-decode kernels; prefill runs in bf16 SDPA. "
        "Three cache variants share the *identical* prefill+greedy-decode loop — only the "
        "decode-step KV dtype/kernel differ — so the numbers below isolate quantization. Each "
        "variant ran in its own process (8 GB host-RAM rule); fp16 is the reference.\n")
    lines.append(f"- Model: `{MODEL}` (bf16), batch=1, single sequence, contiguous blocks, greedy.")
    lines.append(f"- Decode length: {max_new} tokens/prompt; {n} prompts (chat + code).")
    lines.append(f"- Kernel: `paged_decode_attn_splitk{{,_int8}}`, head_dim 128, GQA-6 (12 Q / 2 KV).\n")

    lines.append("## Results (medians across prompts)\n")
    lines.append("| variant | TTFT (ms) | TPOT (ms) | greedy-match vs fp16 | ppl (fp16 cont.) | KV @2048 tok (MiB) | KV ratio |")
    lines.append("|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        s = summary[v]
        lines.append(
            f"| {v} | {s['ttft_ms']:.1f} | {s['tpot_ms']:.2f} | {s['greedy_match']:.3f} | "
            f"{s['ppl']:.3f} | {mb(s['kv_bytes_ref']):.1f} | {s['kv_ratio_vs_fp16']:.3f} |")
    lines.append("")

    pt = summary["int8_per_token"]
    halving = 1.0 - pt["kv_ratio_vs_fp16"]
    lines.append("## Headline\n")
    lines.append(
        f"- **Memory:** INT8 (per-token) KV cache is **{pt['kv_ratio_vs_fp16']:.3f}×** "
        f"the FP16 cache at {ref_len} tokens — a **{100 * halving:.1f}%** reduction (int8 data + fp32 "
        "per-token scales). This number is real and independent of the kernel's latency.")
    lines.append(
        f"- **Precision:** per-token greedy-match vs fp16 = {pt['greedy_match']:.3f}; "
        f"per-tensor = {summary['int8_per_tensor']['greedy_match']:.3f}. Per-token ppl "
        f"{pt['ppl']:.3f} vs fp16 {summary['fp16']['ppl']:.3f}.")
    lines.append("")

    lines.append("## Caveats (do not hide these)\n")
    lines.append("- **Not in vLLM.** Standalone harness; not comparable to the §1–§5 serving matrix; batch=1 latency only.")
    lines.append("- **No fused write kernel.** Per-step KV quantization is torch on the write path, which "
                 "inflates TPOT — the INT8 latency here is *conservative* vs an idealized fused-write impl. "
                 "The memory number is the real headline.")
    lines.append("- **Q is fp16 at the kernel boundary** (kernel is `__half`), not bf16 — a minor precision caveat.")
    lines.append("- **Prefill runs in bf16/SDPA**; only the decode path exercises the custom kernel. The fp16 "
                 "variant matches HF greedy token-for-token (test gate 2), validating the cache+loop before INT8.")
    lines.append("")

    path = os.path.join(_REPO, "report", "int8_bridge.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("wrote", path)


if __name__ == "__main__":
    main()
