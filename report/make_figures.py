"""Generate the figures embedded in report/benchmark_report.md.

Self-contained: kernel/INT8 numbers are the finalized values from
kernel_optimization.md / int8_quant.md / int8_bridge_results.json; spec-decode
points are read from report/specdec_results.csv. Run from anywhere:

    python report/make_figures.py

Writes PNGs into report/figures/.
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

PEAK_BW = 1555.0  # A100-80GB HBM2e peak GB/s
STAGES = ["naive", "vec", "online", "warp", "splitk"]
STAGE_COLOR = "#3b6fb6"


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, HERE))


# ---------------------------------------------------------------- 1. opt ladder
def fig_opt_ladder():
    # Effective GB/s per stage, two regimes that tell the real story:
    #  - batch=1 ctx=4096 (latency-bound): split-K is decisive.
    #  - batch=64 ctx=4096 (bandwidth-bound): online/warp saturate HBM.
    low = [16, 72, 65, 64, 195]            # b=1  ctx=4096 8/8
    high = [425, 1462, 1569, 1591, 1534]   # b=64 ctx=4096 8/8

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, data, title in (
        (axes[0], low, "batch=1, ctx=4096 (latency-bound)"),
        (axes[1], high, "batch=64, ctx=4096 (bandwidth-bound)"),
    ):
        bars = ax.bar(STAGES, data, color=STAGE_COLOR)
        ax.axhline(PEAK_BW, ls="--", lw=1, color="#aa3333")
        ax.text(0.02, PEAK_BW * 0.97, "A100 peak 1555 GB/s",
                color="#aa3333", fontsize=8, va="top", transform=ax.get_yaxis_transform())
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("effective GB/s")
        ax.set_ylim(0, PEAK_BW * 1.08)
        for b, v in zip(bars, data):
            ax.text(b.get_x() + b.get_width() / 2, v + PEAK_BW * 0.01,
                    str(v), ha="center", va="bottom", fontsize=8)
    fig.suptitle("PagedAttention decode — optimization ladder (effective bandwidth)", fontsize=12)
    save(fig, "opt_ladder.png")


# ---------------------------------------------- 2. bandwidth vs batch (roofline)
def fig_bandwidth_roofline():
    # %peak HBM reached vs batch at ctx=4096 (MHA 8/8); kernels are DRAM-bound,
    # so they climb toward the 1555 GB/s ceiling as batch hides launch/latency.
    batch = [1, 8, 32, 64]
    series = {
        "naive":   [1, 6, 18, 27],
        "warp":    [4, 18, 63, 102],
        "splitk":  [13, 60, 86, 99],
        "vllm_v1": [10, 51, 103, 100],
        "vllm_v2": [3, 23, 100, 100],
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, ys in series.items():
        style = "--o" if name.startswith("vllm") else "-o"
        ax.plot(batch, ys, style, label=name, lw=1.8, ms=5)
    ax.axhline(100, ls=":", color="#aa3333", lw=1)
    ax.axhspan(70, 105, color="#dff0d8", alpha=0.5, zorder=0)
    ax.text(1, 72, "70%+ target band", fontsize=8, color="#3a6b35", va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batch)
    ax.set_xticklabels(batch)
    ax.set_xlabel("batch size")
    ax.set_ylabel("% of A100 peak HBM bandwidth")
    ax.set_title("Memory-bound confirmation — ctx=4096, heads 8/8")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    save(fig, "bandwidth_roofline.png")


# ------------------------------------------------------ 3. INT8 precision curves
def fig_int8_precision():
    cfgs = ["ctx128\n8/8", "ctx500\n8/8", "ctx500\n8/2", "ctx2048\n16/4"]
    x = range(len(cfgs))
    pt_max = [2.4261, 1.1912, 1.1582, 1.0009]
    pt_mean = [0.0365, 0.0292, 0.0301, 0.0277]
    ten_max = [2.6207, 2.1629, 2.2581, 1.2929]
    ten_mean = [0.0491, 0.0531, 0.0505, 0.0473]

    fig, (axm, axx) = plt.subplots(1, 2, figsize=(11, 4.2))
    # mean (the representative metric) — per-token consistently below per-tensor
    axm.plot(x, pt_mean, "-o", label="per-token (mean)", color="#2a7", lw=2)
    axm.plot(x, ten_mean, "-s", label="per-tensor (mean)", color="#c63", lw=2)
    axm.set_title("Mean relative error vs FP16 (lower = better)", fontsize=10)
    axm.set_ylabel("mean rel-err")
    # max (noisy: inflated by near-zero denominators)
    axx.plot(x, pt_max, "--o", label="per-token (max)", color="#2a7", lw=1.6)
    axx.plot(x, ten_max, "--s", label="per-tensor (max)", color="#c63", lw=1.6)
    axx.set_title("Max relative error (noise-dominated)", fontsize=10)
    axx.set_ylabel("max rel-err")
    for ax in (axm, axx):
        ax.set_xticks(list(x))
        ax.set_xticklabels(cfgs, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("INT8 KV quantization — per-token vs per-tensor ablation", fontsize=12)
    save(fig, "int8_precision.png")


# --------------------------------------------------- 4. INT8 memory + bridge ppl
def fig_int8_bridge():
    with open(os.path.join(HERE, "int8_bridge_results.json")) as f:
        s = json.load(f)["summary"]
    variants = ["fp16", "int8_per_token", "int8_per_tensor"]
    labels = ["FP16", "INT8\nper-token", "INT8\nper-tensor"]
    ratio = [s[v]["kv_ratio_vs_fp16"] for v in variants]
    ppl = [s[v]["ppl"] for v in variants]

    fig, (axr, axp) = plt.subplots(1, 2, figsize=(10, 4.0))
    bars = axr.bar(labels, ratio, color=["#888", "#2a7", "#c63"])
    axr.axhline(0.5, ls="--", color="#aa3333", lw=1)
    axr.text(2.4, 0.5, "0.5× floor", color="#aa3333", fontsize=8, va="bottom", ha="right")
    axr.set_ylabel("KV-cache size vs FP16")
    axr.set_title("Memory: INT8 KV ≈ 0.52× FP16", fontsize=10)
    for b, v in zip(bars, ratio):
        axr.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8)

    bars = axp.bar(labels, ppl, color=["#888", "#2a7", "#c63"])
    axp.set_ylim(min(ppl) - 0.01, max(ppl) + 0.01)
    axp.set_ylabel("perplexity (FP16 continuation)")
    axp.set_title("Quality: perplexity ~unchanged", fontsize=10)
    for b, v in zip(bars, ppl):
        axp.text(b.get_x() + b.get_width() / 2, v + 0.0005, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8)
    fig.suptitle("End-to-end INT8 bridge — Qwen2.5-1.5B, batch=1, A100", fontsize=12)
    save(fig, "int8_bridge.png")


# --------------------------------------------------------- 5. spec-decode matrix
def _load_specdec():
    rows = []
    with open(os.path.join(HERE, "specdec_results.csv")) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def fig_specdec():
    rows = _load_specdec()

    def pick(method, kv, dataset, load, field, st=None):
        for r in rows:
            if (r["method"] == method and r["kv_dtype"] == kv
                    and r["dataset"] == dataset and r["load"] == str(load)
                    and (st is None or r["num_spec_tokens"] == str(st))):
                v = r[field]
                return float(v) if v else None
        return None

    sts = [2, 4, 8]
    fig, (axa, axt) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) acceptance vs num_spec_tokens (sharegpt, load 8): falls as st grows;
    #     fp8 KV raises acceptance.
    for kv, c in (("auto", "#3b6fb6"), ("fp8", "#c63")):
        acc = [pick("ngram", kv, "sharegpt", 8, "accept_%", st) for st in sts]
        axa.plot(sts, acc, "-o", color=c, lw=2, label=f"ngram, KV={kv}")
    axa.set_xticks(sts)
    axa.set_xlabel("num speculative tokens")
    axa.set_ylabel("draft acceptance %")
    axa.set_title("Acceptance falls as spec length grows\n(sharegpt, concurrency 8)", fontsize=10)
    axa.legend(fontsize=8)
    axa.grid(alpha=0.3)

    # (b) throughput vs load: baseline vs best ngram (st=8 auto) — spec can hurt
    #     once the batch is already busy.
    loads = [1, 8, 32]
    base = [pick("baseline", "auto", "sharegpt", L, "out_tok_s") for L in loads]
    ng = [pick("ngram", "auto", "sharegpt", L, "out_tok_s", 8) for L in loads]
    width = 0.35
    xs = range(len(loads))
    axt.bar([x - width / 2 for x in xs], base, width, label="baseline", color="#888")
    axt.bar([x + width / 2 for x in xs], ng, width, label="ngram st=8 (auto)", color="#2a7")
    axt.set_xticks(list(xs))
    axt.set_xticklabels([f"conc={L}" for L in loads])
    axt.set_ylabel("output tok/s")
    axt.set_title("Throughput: ngram vs baseline\n(sharegpt, KV=auto)", fontsize=10)
    axt.legend(fontsize=8)
    axt.grid(alpha=0.3, axis="y")

    fig.suptitle("vLLM speculative decoding — Qwen2.5-1.5B, A100", fontsize=12)
    save(fig, "specdec.png")


if __name__ == "__main__":
    fig_opt_ladder()
    fig_bandwidth_roofline()
    fig_int8_precision()
    fig_int8_bridge()
    fig_specdec()
    print("done ->", os.path.relpath(FIG, HERE))
