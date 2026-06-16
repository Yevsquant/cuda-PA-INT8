"""Per-layer paged K/V cache for the standalone Qwen INT8 decode bridge.

Scope is deliberately narrow (see thoughts/specdec-benchmark-plan.md §7): **batch=1,
a single sequence, contiguous physical blocks**. That lets the block table be a
fixed identity (`block_table[0, j] = j`) and the per-layer state collapse to one
running length. The cache stores K/V in the exact vLLM/​kernel layout so the custom
kernels (`paged_decode_attn_splitk{,_int8}`) read it directly:

    K: [num_layers, num_blocks, num_kv_heads, head_dim/x, block_size, x]
    V: [num_layers, num_blocks, num_kv_heads, head_dim,   block_size]

Three variants, selected at construction:
    "fp16"            — fp16 cache, FP16 kernel (the non-quantized baseline)
    "int8_per_token"  — int8 cache + one fp32 scale per (token, kv_head)
    "int8_per_tensor" — int8 cache + one fp32 scale per layer (static, ablation)

INT8 scale buffers follow the kernel layout [num_layers, num_blocks, num_kv_heads,
block_size]. Per-tensor reuses a single layer-global scalar (computed once at
prefill) broadcast into every slot, so the *same* int8 kernel serves both modes.
"""

import os
import sys

import torch

# Reuse the per-token quant used by the kernel tests (established cross-import
# pattern: bench_paged_decode.py does the same).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "tests"))
from paged_decode_attn import quantize_per_token  # noqa: E402


def _quantize_with_scale(x, scale):
    """Symmetric int8 quant of `x` using a precomputed `scale` (scalar or
    broadcastable). Mirrors quantize_per_token's rounding/clamping."""
    q = torch.round(x.to(torch.float32) / scale).clamp(-127, 127).to(torch.int8)
    return q


class PagedKVCache:
    def __init__(self, num_layers, num_kv_heads, head_dim, max_len,
                 block_size=16, x=8, variant="fp16", device="cuda"):
        assert variant in ("fp16", "int8_per_token", "int8_per_tensor")
        assert head_dim % x == 0
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.x = x
        self.variant = variant
        self.is_int8 = variant != "fp16"
        self.device = device

        self.num_blocks = (max_len + block_size - 1) // block_size
        store_dtype = torch.int8 if self.is_int8 else torch.float16

        self.k_cache = torch.zeros(
            num_layers, self.num_blocks, num_kv_heads, head_dim // x, block_size, x,
            dtype=store_dtype, device=device)
        self.v_cache = torch.zeros(
            num_layers, self.num_blocks, num_kv_heads, head_dim, block_size,
            dtype=store_dtype, device=device)
        if self.is_int8:
            self.k_scales = torch.zeros(
                num_layers, self.num_blocks, num_kv_heads, block_size,
                dtype=torch.float32, device=device)
            self.v_scales = torch.zeros(
                num_layers, self.num_blocks, num_kv_heads, block_size,
                dtype=torch.float32, device=device)
            # Layer-global static scales for the per-tensor ablation.
            self._k_tensor_scale = [None] * num_layers
            self._v_tensor_scale = [None] * num_layers

        # Contiguous identity block table, shared across layers (batch=1).
        self.block_table = torch.arange(
            self.num_blocks, dtype=torch.int32, device=device).view(1, -1)
        self.lengths = [0] * num_layers

    def reset(self):
        self.lengths = [0] * self.num_layers
        if self.is_int8:
            self._k_tensor_scale = [None] * self.num_layers
            self._v_tensor_scale = [None] * self.num_layers

    # --- write path ---------------------------------------------------------

    def prefill(self, layer_idx, k, v):
        """Scatter the prompt's K/V (shape [L, num_kv_heads, head_dim]) into
        positions [0, L) of `layer_idx`. Sets the layer length to L."""
        L, H, D = k.shape
        assert H == self.num_kv_heads and D == self.head_dim
        nb = (L + self.block_size - 1) // self.block_size
        P = nb * self.block_size
        x, bs = self.x, self.block_size

        if self.variant == "int8_per_token":
            kq, ks = quantize_per_token(k, dim=-1)   # kq [L,H,D], ks [L,H]
            vq, vs = quantize_per_token(v, dim=-1)
        elif self.variant == "int8_per_tensor":
            sk = (k.abs().amax() / 127.0).to(torch.float32).clamp_min(1e-8)
            sv = (v.abs().amax() / 127.0).to(torch.float32).clamp_min(1e-8)
            self._k_tensor_scale[layer_idx] = sk
            self._v_tensor_scale[layer_idx] = sv
            kq = _quantize_with_scale(k, sk)
            vq = _quantize_with_scale(v, sv)
            ks = sk.expand(L, H)
            vs = sv.expand(L, H)
        else:  # fp16
            kq, vq = k.to(torch.float16), v.to(torch.float16)

        # K -> [nb, H, D/x, bs, x]
        kp = torch.zeros(P, H, D, dtype=kq.dtype, device=self.device)
        kp[:L] = kq
        self.k_cache[layer_idx, :nb] = (
            kp.view(nb, bs, H, D // x, x).permute(0, 2, 3, 1, 4).contiguous())
        # V -> [nb, H, D, bs]
        vp = torch.zeros(P, H, D, dtype=vq.dtype, device=self.device)
        vp[:L] = vq
        self.v_cache[layer_idx, :nb] = (
            vp.view(nb, bs, H, D).permute(0, 2, 3, 1).contiguous())

        if self.is_int8:
            ksp = torch.zeros(P, H, dtype=torch.float32, device=self.device)
            vsp = torch.zeros(P, H, dtype=torch.float32, device=self.device)
            ksp[:L], vsp[:L] = ks, vs
            self.k_scales[layer_idx, :nb] = ksp.view(nb, bs, H).permute(0, 2, 1).contiguous()
            self.v_scales[layer_idx, :nb] = vsp.view(nb, bs, H).permute(0, 2, 1).contiguous()

        self.lengths[layer_idx] = L

    def decode_append(self, layer_idx, k, v):
        """Append one decoded token's K/V (shape [num_kv_heads, head_dim]) at the
        layer's current length. Returns the new length (= kernel context_len)."""
        t = self.lengths[layer_idx]
        blk, off = t // self.block_size, t % self.block_size
        H, D, x = self.num_kv_heads, self.head_dim, self.x

        if self.variant == "int8_per_token":
            kq, ks = quantize_per_token(k, dim=-1)   # kq [H,D], ks [H]
            vq, vs = quantize_per_token(v, dim=-1)
        elif self.variant == "int8_per_tensor":
            sk, sv = self._k_tensor_scale[layer_idx], self._v_tensor_scale[layer_idx]
            kq, vq = _quantize_with_scale(k, sk), _quantize_with_scale(v, sv)
            ks, vs = sk.expand(H), sv.expand(H)
        else:
            kq, vq = k.to(torch.float16), v.to(torch.float16)

        self.k_cache[layer_idx, blk, :, :, off, :] = kq.view(H, D // x, x)
        self.v_cache[layer_idx, blk, :, :, off] = vq
        if self.is_int8:
            self.k_scales[layer_idx, blk, :, off] = ks
            self.v_scales[layer_idx, blk, :, off] = vs

        self.lengths[layer_idx] = t + 1
        return t + 1

    # --- read path ----------------------------------------------------------

    def context_lens(self, layer_idx):
        return torch.tensor([self.lengths[layer_idx]], dtype=torch.int32, device=self.device)

    def kv_bytes(self):
        """Total bytes the K/V cache (and scales) currently occupy — the headline
        memory number. Counts the full allocation (num_blocks)."""
        b = self.k_cache.element_size() * self.k_cache.nelement()
        b += self.v_cache.element_size() * self.v_cache.nelement()
        if self.is_int8:
            b += self.k_scales.element_size() * self.k_scales.nelement()
            b += self.v_scales.element_size() * self.v_scales.nelement()
        return b

    def kv_bytes_for_len(self, n):
        """Bytes the K+V cache (and int8 scales) need for `n` tokens, analytically
        (blocks = ceil(n/block_size)), across all layers. Geometry-clean headline."""
        blocks = (n + self.block_size - 1) // self.block_size
        elem = 1 if self.is_int8 else 2
        per_blk = 2 * self.num_kv_heads * self.head_dim * self.block_size  # K + V
        b = blocks * per_blk * elem * self.num_layers
        if self.is_int8:
            scale_per_blk = 2 * self.num_kv_heads * self.block_size  # k + v scales
            b += blocks * scale_per_blk * 4 * self.num_layers  # fp32
        return b
