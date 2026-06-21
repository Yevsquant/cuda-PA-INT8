"""RBBQ Phase 4 gate — weight/act/group decomposition of W8A8 damage on Qwen2.5.

The BERT diagnostic (Phase 3) found the C1 damage is activation-granularity at the
update-branch OUTPUT linears (out_proj/fc2), non-foldable, branch-irrelevant.
Decoder LLMs may differ: Qwen's massive activations live in the residual stream
feeding the LN->qkv/gate-up path, which IS foldable and branch-relevant.

This localizes Qwen2.5-1.5B's C1 (static per-tensor W8A8) damage by group, then
splits the dominant group into weight vs activation, using WikiText-2 perplexity.

Groups (per decoder layer):
  attn_in  = q_proj,k_proj,v_proj   (LN-fed, foldable)
  attn_out = o_proj                 (update-branch output)
  mlp_in   = gate_proj,up_proj      (LN-fed, foldable)
  mlp_out  = down_proj              (update-branch output)
"""
import json, math
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from phase3_variant_b import FlexQuantLinear

DEV = "cuda"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SUF = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
       "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
GROUP = {"self_attn.q_proj": "attn_in", "self_attn.k_proj": "attn_in",
         "self_attn.v_proj": "attn_in", "self_attn.o_proj": "attn_out",
         "mlp.gate_proj": "mlp_in", "mlp.up_proj": "mlp_in",
         "mlp.down_proj": "mlp_out"}


def group_of(name):
    for s in SUF:
        if name.endswith(s):
            return GROUP[s]
    return None


def targets(model):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and group_of(name):
            yield name, mod


def fresh():
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype="bfloat16").to(DEV).eval()


def wrap(model, spec):
    qlins = []
    for name, mod in list(targets(model)):
        opt = spec(name)
        if opt is None:
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], FlexQuantLinear(mod, **opt))
        qlins.append(model.get_submodule(name))
    return qlins


@torch.no_grad()
def calibrate(model, qlins, ids, seqlen=2048, n=8):
    for ql in qlins:
        if ql.qa:
            ql.calibrating = True
    for i in range(n):
        model(ids[:, i * seqlen:(i + 1) * seqlen])
    for ql in qlins:
        ql.calibrating = False


@torch.no_grad()
def ppl(model, ids, seqlen=2048, n=30):
    nll, tok = 0.0, 0
    for i in range(n):
        b = ids[:, i * seqlen:(i + 1) * seqlen]
        if b.shape[1] < seqlen:
            break
        loss = model(b, labels=b).loss
        nll += loss.item() * (seqlen - 1)
        tok += seqlen - 1
    return math.exp(nll / tok)


def run(spec, calib_ids, eval_ids):
    model = fresh()
    qlins = wrap(model, spec)
    calibrate(model, qlins, calib_ids)
    p = ppl(model, eval_ids)
    del model; torch.cuda.empty_cache()
    return p


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    cal = tok("\n\n".join(t for t in ds["train"]["text"] if t.strip()),
              return_tensors="pt").input_ids.to(DEV)
    ev = tok("\n\n".join(t for t in ds["test"]["text"] if t.strip()),
             return_tensors="pt").input_ids.to(DEV)

    C1 = dict(qw=True, qa=True, w_gran="tensor")
    keep = lambda g: (lambda n: None if group_of(n) == g else dict(C1))
    only = lambda g, o: (lambda n: dict(o) if group_of(n) == g else dict(C1))
    specs = {
        "plain":        lambda n: dict(C1),
        "fp16_attn_in":  keep("attn_in"),
        "fp16_attn_out": keep("attn_out"),
        "fp16_mlp_in":   keep("mlp_in"),
        "fp16_mlp_out":  keep("mlp_out"),
        # weight/act split on the dominant group (down_proj = mlp_out)
        "mlp_out_w":   only("mlp_out", dict(qw=False, qa=True, w_gran="tensor")),  # FP16 weight, quant act
        "mlp_out_a":   only("mlp_out", dict(qw=True, qa=False, w_gran="tensor")),  # quant weight, FP16 act
        "mlp_out_dyn": only("mlp_out", dict(qw=True, qa=True, w_gran="channel", act_mode="dynamic")),  # the fix
    }
    fp = ppl(fresh(), ev)
    print(f"=== Qwen2.5-1.5B WikiText-2 ppl (FP bf16 = {fp:.3f}) ===")
    res = {"fp": round(fp, 3)}
    for k, sp in specs.items():
        res[k] = round(run(sp, cal, ev), 3)
        print(f"  {k:14s} ppl={res[k]:.3f}  (dppl={res[k]-fp:+.3f})")
    json.dump(res, open("report/phase4_qwen_localize.json", "w"), indent=2)
    print("wrote report/phase4_qwen_localize.json")


if __name__ == "__main__":
    main()
