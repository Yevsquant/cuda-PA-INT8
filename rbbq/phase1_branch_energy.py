"""RBBQ Phase 1 — residual-branch energy instrumentation (the motivation GATE).

At every residual add `y = identity + update`, measure per-channel energy of each
branch and the imbalance r[c] = E_id[c]/E_up[c]. Gate: is r[c] heavy-tailed (a
few channels strongly dominated by one branch)? If flat, the RBBQ premise fails.

Covers both folding regimes from rbbq_method.md:
  - BERT-base  : post-norm LayerNorm  (add then norm)
  - Qwen2.5-1.5B: pre-norm  RMSNorm   (norm then add)  [Llama-family proxy]

Also reports two descriptive outlier metrics per add:
  - channel dominance  : max_c E_id[c] / median_c E_id[c]  (Bondarenko claim)
  - crush rate         : fraction of |add output| that quantizes to |level|<=1
                         under per-tensor int8 max-scaling (outlier-induced
                         resolution loss on the quiet bulk)
"""
import json, os
import numpy as np
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForCausalLM)
from datasets import load_dataset

DEV = "cuda"
QMAX = 127.0
FIGDIR = "report/figs"
CUR_MASK = None  # [N_valid_tokens] index into flattened [B*T] for current batch


class BranchStats:
    """Accumulates per-channel sum of squares for identity/update at each add."""
    def __init__(self, d):
        self.d = d
        self.s = {}  # key (layer, add) -> dict of running tensors

    def _slot(self, key):
        if key not in self.s:
            self.s[key] = dict(ssq_id=torch.zeros(self.d, device=DEV),
                               ssq_up=torch.zeros(self.d, device=DEV),
                               n=0, crush=0.0, crush_n=0)
        return self.s[key]

    def add(self, key, idt, upt):
        sl = self._slot(key)
        idf = idt.reshape(-1, self.d).float()
        upf = upt.reshape(-1, self.d).float()
        if CUR_MASK is not None:
            idf, upf = idf[CUR_MASK], upf[CUR_MASK]
        sl["ssq_id"] += (idf * idf).sum(0)
        sl["ssq_up"] += (upf * upf).sum(0)
        sl["n"] += idf.shape[0]
        o = idf + upf
        scale = o.abs().max().clamp_min(1e-8) / QMAX
        lvl = torch.round(o / scale).abs()
        sl["crush"] += (lvl <= 1).float().mean().item()
        sl["crush_n"] += 1

    def summary(self):
        out = {}
        for (layer, add), sl in sorted(self.s.items()):
            e_id = (sl["ssq_id"] / sl["n"]).cpu().numpy()
            e_up = (sl["ssq_up"] / sl["n"]).cpu().numpy()
            r = e_id / np.clip(e_up, 1e-12, None)
            log2r = np.log2(np.clip(r, 1e-12, None))
            out[f"L{layer}_add{add}"] = dict(
                log2r=log2r,
                frac_imbalanced=float(np.mean(np.abs(log2r) > 2)),   # >2 octaves
                spread_log2r=float(np.std(log2r)),
                dom_id=float(e_id.max() / np.median(e_id)),
                dom_up=float(e_up.max() / np.median(e_up)),
                crush=sl["crush"] / sl["crush_n"],
            )
        return out


def hook_bert(model, stats):
    enc = model.bert.encoder.layer
    tmp = {}
    for i, layer in enumerate(enc):
        so, oo = layer.attention.output, layer.output
        so.register_forward_pre_hook(lambda m, a, i=i: tmp.__setitem__(f"{i}0", a[1]))
        so.dense.register_forward_hook(lambda m, i_, o, i=i: tmp.__setitem__(f"{i}0_u", o))
        so.register_forward_hook(lambda m, i_, o, i=i: stats.add((i, 0), tmp[f"{i}0"], tmp[f"{i}0_u"]))
        oo.register_forward_pre_hook(lambda m, a, i=i: tmp.__setitem__(f"{i}1", a[1]))
        oo.dense.register_forward_hook(lambda m, i_, o, i=i: tmp.__setitem__(f"{i}1_u", o))
        oo.register_forward_hook(lambda m, i_, o, i=i: stats.add((i, 1), tmp[f"{i}1"], tmp[f"{i}1_u"]))


def hook_qwen(model, stats):
    layers = model.model.layers
    tmp = {}
    for i, layer in enumerate(layers):
        layer.register_forward_pre_hook(lambda m, a, i=i: tmp.__setitem__(f"{i}x", a[0]))
        layer.self_attn.o_proj.register_forward_hook(lambda m, i_, o, i=i: tmp.__setitem__(f"{i}u0", o))
        layer.mlp.down_proj.register_forward_hook(lambda m, i_, o, i=i: tmp.__setitem__(f"{i}u1", o))
        def post(m, i_, o, i=i):
            x, u0, u1 = tmp[f"{i}x"], tmp[f"{i}u0"], tmp[f"{i}u1"]
            stats.add((i, 0), x, u0)
            stats.add((i, 1), x + u0, u1)
        layer.register_forward_hook(post)


@torch.no_grad()
def run_bert(n=512):
    global CUR_MASK
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    model = AutoModelForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-SST-2").to(DEV).eval()
    stats = BranchStats(model.config.hidden_size)
    hook_bert(model, stats)
    rows = load_dataset("nyu-mll/glue", "sst2")["train"][:n]
    for i in range(0, n, 32):
        enc = tok(rows["sentence"][i:i + 32], padding=True, truncation=True,
                  max_length=128, return_tensors="pt").to(DEV)
        CUR_MASK = enc["attention_mask"].reshape(-1).bool()
        model(**enc)
    CUR_MASK = None
    return stats.summary()


@torch.no_grad()
def run_qwen(n_seq=128, seqlen=256):
    global CUR_MASK
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", dtype="bfloat16").to(DEV).eval()
    stats = BranchStats(model.config.hidden_size)
    hook_qwen(model, stats)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["train"]
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    CUR_MASK = None  # no padding, all tokens valid
    for k in range(n_seq):
        chunk = ids[k * seqlen:(k + 1) * seqlen]
        if chunk.numel() < seqlen:
            break
        model(chunk.unsqueeze(0).to(DEV))
    return stats.summary()


def aggregate(summary):
    keys = list(summary)
    return dict(
        n_adds=len(keys),
        frac_imbalanced=float(np.mean([summary[k]["frac_imbalanced"] for k in keys])),
        spread_log2r=float(np.mean([summary[k]["spread_log2r"] for k in keys])),
        max_dom_id=float(np.max([summary[k]["dom_id"] for k in keys])),
        mean_dom_id=float(np.mean([summary[k]["dom_id"] for k in keys])),
        max_dom_up=float(np.max([summary[k]["dom_up"] for k in keys])),
        mean_crush=float(np.mean([summary[k]["crush"] for k in keys])),
    )


def plot(tag, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGDIR, exist_ok=True)
    keys = list(summary)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    pooled = np.concatenate([summary[k]["log2r"] for k in keys])
    ax[0].hist(np.clip(pooled, -10, 10), bins=80, color="steelblue")
    ax[0].set_title(f"{tag}: log2(E_id/E_up) over all channels/adds")
    ax[0].set_xlabel("log2 r[c]"); ax[0].axvline(0, color="k", lw=0.5)
    ax[1].plot([summary[k]["dom_id"] for k in keys], label="identity")
    ax[1].plot([summary[k]["dom_up"] for k in keys], label="update")
    ax[1].set_title("channel dominance (max/median E)"); ax[1].set_xlabel("add index")
    ax[1].set_yscale("log"); ax[1].legend()
    ax[2].plot([summary[k]["crush"] for k in keys], color="firebrick")
    ax[2].set_title("crush rate (|level|<=1 frac)"); ax[2].set_xlabel("add index")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/phase1_{tag}.png", dpi=90)
    print("wrote", f"{FIGDIR}/phase1_{tag}.png")


def main():
    out = {}
    for tag, fn in [("bert", run_bert), ("qwen", run_qwen)]:
        print(f"=== {tag} ===")
        summ = fn()
        plot(tag, summ)
        agg = aggregate(summ)
        out[tag] = agg
        # strip raw arrays before JSON
        out[tag + "_perlayer"] = {k: {kk: vv for kk, vv in v.items() if kk != "log2r"}
                                  for k, v in summ.items()}
        print(json.dumps(agg, indent=2))
    with open("report/phase1_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote report/phase1_results.json")


if __name__ == "__main__":
    main()
