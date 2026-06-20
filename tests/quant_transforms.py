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

import torch


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
