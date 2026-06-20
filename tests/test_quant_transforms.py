"""Unit tests for SmoothQuant transform (Phase 2). Pure-torch, CPU.

Two properties matter: (1) the transform is *exact* on the dot product, so it
costs no accuracy by itself; (2) it actually flattens K's per-channel imbalance,
which is what makes K quantize better.
"""

import torch

from paged_decode_attn import quantize_per_token
from quant_metrics import rel_l2
from quant_transforms import apply_smooth, smoothquant_factor


def _channel_absmax(x):
    # x: [tokens, head_dim] -> [head_dim]
    return x.abs().amax(dim=0)


def test_smooth_qk_is_exact_on_dot():
    """Q'·K'ᵀ == Q·Kᵀ to fp tolerance for any factor s."""
    torch.manual_seed(0)
    q = torch.randn(5, 128)   # [tokens_q, hd]
    k = torch.randn(7, 128)
    s = torch.rand(128) * 3 + 0.1
    qs, ks = apply_smooth(q, k, s, s)
    torch.testing.assert_close(qs @ ks.T, q @ k.T, atol=1e-4, rtol=1e-4)


def test_smooth_reduces_channel_imbalance():
    """Injecting an outlier channel into K, SmoothQuant lowers the per-channel
    imbalance (max/median channel absmax) — the whole point."""
    torch.manual_seed(0)
    q = torch.randn(64, 128)
    k = torch.randn(64, 128)
    k[:, 0] *= 100.0  # outlier channel

    s = smoothquant_factor(_channel_absmax(k), _channel_absmax(q), alpha=0.5)
    _, k_smooth = apply_smooth(q, k, s, s)

    def imbalance(x):
        c = _channel_absmax(x)
        return (c.amax() / c.median()).item()

    assert imbalance(k_smooth) < imbalance(k)


def test_smooth_improves_per_token_quant_with_outlier():
    """End goal: with an outlier channel, smoothing K before per-token quant cuts
    the round-trip error on the normal channels."""
    torch.manual_seed(0)
    q = torch.randn(128, 128)
    k = torch.randn(128, 128)
    k[:, 0] = 80.0  # systematic outlier channel

    s = smoothquant_factor(_channel_absmax(k), _channel_absmax(q), alpha=0.5)
    _, k_smooth = apply_smooth(q, k, s, s)

    def normal_channel_err(x):
        qi, sc = quantize_per_token(x, dim=-1)
        dq = qi.float() * sc.unsqueeze(-1)
        return rel_l2(dq[:, 1:], x[:, 1:], dim=-1).mean().item()

    assert normal_channel_err(k_smooth) < 0.5 * normal_channel_err(k)
