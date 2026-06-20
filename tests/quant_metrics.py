"""Output-error metrics for INT8-vs-FP16 attention comparison.

The old test measured elementwise |out_int8 - out_fp16| / |out_fp16|, whose max
explodes wherever an fp16 output element is near zero — so the reported "max
error" was mostly noise. These two metrics look at the *whole vector* instead:

  - cosine_sim: direction agreement, immune to per-element zeros.
  - rel_l2:     ||a - b|| / ||b||, a single well-defined relative magnitude.

Both reduce over `dim` (head_dim by default), returning one value per remaining
(seq, head) slot; callers aggregate with .mean()/.min()/.max().
"""

import torch


def cosine_sim(a, b, dim=-1, eps=1e-12):
    """Cosine similarity between `a` and `b` along `dim`."""
    a = a.float()
    b = b.float()
    num = (a * b).sum(dim=dim)
    den = (a.norm(dim=dim) * b.norm(dim=dim)).clamp_min(eps)
    return num / den


def rel_l2(a, b, dim=-1, eps=1e-12):
    """Norm-normalized error ||a - b|| / ||b|| along `dim`. `b` is the reference."""
    a = a.float()
    b = b.float()
    return (a - b).norm(dim=dim) / b.norm(dim=dim).clamp_min(eps)
