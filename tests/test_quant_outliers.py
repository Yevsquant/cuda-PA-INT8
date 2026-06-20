"""Outlier / heavy-tail stress tests for symmetric per-token INT8 quant.

These are *characterization* tests, not pass/fail theater: they document the
known weakness — symmetric INT8 with scale = max(|x|)/127 degrades as a single
channel's magnitude grows, because the outlier inflates the scale and starves
the normal values of resolution. The asserts pin the *direction* of that
degradation so a future fix (SmoothQuant / Hadamard / clipping) can show up as
the trend flattening. Pure-torch, runs on CPU.
"""

import torch

from paged_decode_attn import quantize_per_token
from quant_metrics import rel_l2


def _roundtrip_rel_l2(x):
    q, s = quantize_per_token(x, dim=-1)
    dq = q.float() * s.unsqueeze(-1)
    return rel_l2(dq, x, dim=-1)


def test_heavy_tail_worse_than_gaussian():
    """Student-t (heavy-tailed) KV round-trips worse than N(0,1)."""
    torch.manual_seed(0)
    shape = (256, 128)
    gauss = torch.randn(*shape)
    # Student-t with 3 dof: same center, much heavier tails.
    t3 = torch.distributions.StudentT(df=3.0).sample(shape)

    assert _roundtrip_rel_l2(t3).mean() > _roundtrip_rel_l2(gauss).mean()


def test_channel_outlier_starves_normal_channels():
    """A single large-magnitude channel inflates the per-token scale and starves
    the *normal* channels of resolution — the post-RoPE outlier-channel failure
    mode. The damage shows up in the normal channels (which feed the Q·K dot),
    not in whole-vector rel-L2: a huge outlier quantizes near-perfectly and
    dominates the norm, masking the harm. So we measure the normal-channel slice."""
    torch.manual_seed(0)
    base = torch.randn(512, 128)

    def normal_channel_err(mag):
        x = base.clone()
        x[:, 0] = mag  # one systematic outlier channel
        q, s = quantize_per_token(x, dim=-1)
        dq = q.float() * s.unsqueeze(-1)
        return rel_l2(dq[:, 1:], x[:, 1:], dim=-1).mean().item()

    errs = [normal_channel_err(mag) for mag in (1.0, 10.0, 100.0, 1000.0)]

    # Bigger outlier -> coarser scale -> more error on the normal channels.
    for lo, hi in zip(errs, errs[1:]):
        assert hi >= lo - 1e-4, errs
    # The worst case is catastrophically worse than the clean baseline.
    assert errs[-1] > 10.0 * errs[0], errs
