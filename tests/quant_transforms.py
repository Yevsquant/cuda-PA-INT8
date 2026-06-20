"""Outlier-taming transforms applied *before* quantization (Phase 2+).

These fold into Q (fp32, host-side) and a pre-transformed K, leaving the INT8
kernel data path untouched: since the transform is exact in fp, Q'·K'ᵀ = Q·Kᵀ,
so only K's *quantizability* changes, not the attention scores.

SmoothQuant (方案 B): migrate per-channel difficulty from K (quantized) to Q
(stays fp). K' = K / s, Q' = Q * s with a per-channel factor
    s_d = max|K_d|^alpha / max|Q_d|^(1-alpha).
Bigger outlier channels in K get a bigger s, so K' is flatter and the per-token
scale (max over head_dim) no longer wasted on them.
"""

import math

import torch


def hadamard_matrix(n, device=None, dtype=torch.float32):
    """Normalized (orthonormal) Sylvester-Hadamard matrix; `n` a power of 2.

    Built by the Kronecker doubling [[H,H],[H,-H]] then scaled by 1/sqrt(n), so
    entries are ±1/sqrt(n), H is symmetric, and H @ H.T == I (H^-1 == H)."""
    assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"
    H = torch.ones(1, 1, device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


def hadamard_apply(q, k, H):
    """q' = q @ H, k' = k @ H — an orthonormal rotation along head_dim.

    Method 方案 D: spreads each token's outlier energy uniformly across channels
    (gaussianizing the per-token distribution), while q'·k'ᵀ == q·kᵀ since
    H H^T = I — so it's free on the scores and needs no INT8 kernel change.
    Data-independent: no calibration, unlike SmoothQuant."""
    return q @ H, k @ H


def smoothquant_factor(k_absmax, q_absmax, alpha=0.5, eps=1e-5):
    """Per-channel SmoothQuant factor, shape == k_absmax/q_absmax ([..., head_dim]).

    k_absmax, q_absmax: calibration abs-max per channel (K is the quantized side
    to tame; Q absorbs the imbalance). alpha in [0,1] trades how much difficulty
    moves to Q (alpha=1 normalizes K's per-channel range fully onto Q)."""
    k_absmax = k_absmax.clamp_min(eps)
    q_absmax = q_absmax.clamp_min(eps)
    return (k_absmax.pow(alpha) / q_absmax.pow(1.0 - alpha)).clamp_min(eps)


def apply_smooth(q, k, s_q, s_k):
    """q' = q * s_q, k' = k / s_k, broadcasting over the head_dim (last) axis.

    For a matched (q_head, kv_head) pair the two factors are the SAME per-channel
    vector (under GQA, q heads use their kv head's factor) — that is what makes
    sum_d q'[d]·k'[d] == sum_d q[d]·k[d] hold exactly."""
    return q * s_q, k / s_k
