import torch
import math

def paged_decode_attention_reference(
    q,             # [num_seqs, num_heads, head_dim]
    k_blocks,      # [num_blocks, num_kv_heads, head_dim/x, block_size, x]
    v_blocks,      # [num_blocks, num_kv_heads, head_dim, block_size]   (real vLLM layout)
    block_table,   # [num_seqs, max_num_blocks_per_seq]
    context_lens,  # [num_seqs] The current actual effective token length of each seq.
    block_size=16
):
    num_seqs, num_heads, head_dim = q.shape
    num_kv_heads = k_blocks.shape[1]
    queries_per_kv_head = num_heads // num_kv_heads
    out = torch.zeros_like(q)

    for b in range(num_seqs):
        seq_len = context_lens[b].item()
        b_blocks = block_table[b]

        # 1. Collect all Ks and Vs for the current seq
        k_seq, v_seq = [], []
        for i in range(seq_len):
            block_idx = b_blocks[i // block_size].item()
            block_offset = i % block_size

            # k_blocks: [num_blocks, num_kv_heads, head_dim/x, block_size, x]
            k_token = k_blocks[block_idx, :, :, block_offset, :].reshape(num_kv_heads, head_dim)
            # v_blocks: [num_blocks, num_kv_heads, head_dim, block_size]
            v_token = v_blocks[block_idx, :, :, block_offset]

            k_seq.append(k_token)
            v_seq.append(v_token)

        k_seq = torch.stack(k_seq, dim=1) # [num_kv_heads, seq_len, head_dim]
        v_seq = torch.stack(v_seq, dim=0) # [seq_len, num_kv_heads, head_dim]

        # 2. Attention Computation (GQA supported)
        for h in range(num_heads):
            kv_h = h // queries_per_kv_head
            q_head = q[b, h, :] # [head_dim]
            k_head = k_seq[kv_h, :seq_len, :] # [seq_len, head_dim]
            v_head = v_seq[:seq_len, kv_h, :] # [seq_len, head_dim]

            scores = torch.matmul(k_head, q_head) / math.sqrt(head_dim) # [seq_len]
            probs = torch.softmax(scores, dim=-1) # [seq_len]
            out[b, h, :] = torch.sum(probs.unsqueeze(-1) * v_head, dim=0)

    return out


def build_paged_kv_cache(
    k,             # [num_seqs, max_len, num_kv_heads, head_dim]
    v,             # [num_seqs, max_len, num_kv_heads, head_dim]
    context_lens,  # [num_seqs]
    block_size=16,
    x=8,
):
    """Scatter contiguous per-seq K/V into the paged vLLM cache layout.

    Builds a *shuffled* block_table so physical blocks are not contiguous,
    exercising the gather path. Returns (k_cache, v_cache, block_table).
    """
    num_seqs, max_len, num_kv_heads, head_dim = k.shape
    assert head_dim % x == 0
    device, dtype = k.device, k.dtype

    max_blocks_per_seq = (max_len + block_size - 1) // block_size
    blocks_per_seq = [
        (int(context_lens[b]) + block_size - 1) // block_size for b in range(num_seqs)
    ]
    # +1 so the pool has an unused/padding block and ids aren't fully packed.
    num_blocks = sum(blocks_per_seq) + 1
    perm = torch.randperm(num_blocks, device=device)  # shuffled physical block ids

    k_cache = torch.zeros(
        num_blocks, num_kv_heads, head_dim // x, block_size, x, dtype=dtype, device=device
    )
    v_cache = torch.zeros(
        num_blocks, num_kv_heads, head_dim, block_size, dtype=dtype, device=device
    )
    block_table = torch.zeros(num_seqs, max_blocks_per_seq, dtype=torch.int32, device=device)

    next_block = 0
    for b in range(num_seqs):
        clen = int(context_lens[b])
        for j in range(blocks_per_seq[b]):
            phys = int(perm[next_block].item())
            next_block += 1
            block_table[b, j] = phys
            t0 = j * block_size
            for off in range(block_size):
                t = t0 + off
                if t >= clen:
                    break
                # K: [H, D] -> [H, D/x, x] at (phys, :, :, off, :)
                k_cache[phys, :, :, off, :] = k[b, t].view(num_kv_heads, head_dim // x, x)
                # V: [H, D] -> (phys, :, :, off)
                v_cache[phys, :, :, off] = v[b, t]

    return k_cache, v_cache, block_table


# --- INT8 KV-cache quantization (symmetric, s = max(|x|)/127) ---------------

def quantize_per_token(x, dim=-1):
    """Symmetric per-token INT8 quant. `x` is [..., head_dim]; one fp32 scale
    per token (reduced over `dim`). Returns (int8 tensor, fp32 scale)."""
    amax = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8)
    scale = (amax / 127.0).to(torch.float32)
    q = torch.round(x.to(torch.float32) / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(dim)


def quantize_per_tensor(x):
    """Symmetric per-tensor INT8 quant: a single fp32 scalar scale for all of
    `x`. Returns (int8 tensor, fp32 scalar scale)."""
    amax = x.abs().amax().clamp_min(1e-8)
    scale = (amax / 127.0).to(torch.float32)
    q = torch.round(x.to(torch.float32) / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def quantize_per_token_asym(x, dim=-1):
    """Asymmetric per-token INT8 quant. Maps each token's [min,max] over `dim`
    onto the full signed-int8 range [-128,127], so skewed (non-zero-centered)
    distributions don't waste half the range. Returns (int8, fp32 scale, fp32
    zero-point), with dequant x ≈ scale * (q - zero_point).

      s  = (max - min) / 255
      z  = round(-128 - min / s)              # qmin - min/s, qmin = -128
      q  = clamp(round(x/s) + z, -128, 127)
    """
    xmax = x.amax(dim=dim, keepdim=True)
    xmin = x.amin(dim=dim, keepdim=True)
    scale = ((xmax - xmin) / 255.0).clamp_min(1e-8).to(torch.float32)
    zero = torch.round(-128.0 - xmin / scale).to(torch.float32)
    q = torch.round(x.to(torch.float32) / scale + zero).clamp(-128, 127).to(torch.int8)
    return q, scale.squeeze(dim), zero.squeeze(dim)


def build_paged_kv_cache_int8(
    k,             # [num_seqs, max_len, num_kv_heads, head_dim]
    v,             # [num_seqs, max_len, num_kv_heads, head_dim]
    context_lens,  # [num_seqs]
    block_size=16,
    x=8,
    mode="per_token",
):
    """INT8 analogue of build_paged_kv_cache. Mirrors the scatter/shuffle/block-
    table logic exactly, but stores int8 K/V plus matching fp32 scale buffers.

    Scales are per (physical-block, kv_head, token-in-block), shaped
    [num_blocks, num_kv_heads, block_size]:
      - per_token: each token×kv_head gets its own scale (reduced over head_dim).
      - per_tensor: one scalar scale for the whole K tensor and one for V, then
        broadcast into every scale slot (the ablation baseline).

    Returns (k_cache:int8, v_cache:int8, k_scales:fp32, v_scales:fp32,
             block_table).
    """
    assert mode in ("per_token", "per_tensor")
    num_seqs, max_len, num_kv_heads, head_dim = k.shape
    assert head_dim % x == 0
    device = k.device

    max_blocks_per_seq = (max_len + block_size - 1) // block_size
    blocks_per_seq = [
        (int(context_lens[b]) + block_size - 1) // block_size for b in range(num_seqs)
    ]
    num_blocks = sum(blocks_per_seq) + 1
    perm = torch.randperm(num_blocks, device=device)

    k_cache = torch.zeros(
        num_blocks, num_kv_heads, head_dim // x, block_size, x,
        dtype=torch.int8, device=device
    )
    v_cache = torch.zeros(
        num_blocks, num_kv_heads, head_dim, block_size, dtype=torch.int8, device=device
    )
    k_scales = torch.zeros(num_blocks, num_kv_heads, block_size, dtype=torch.float32, device=device)
    v_scales = torch.zeros(num_blocks, num_kv_heads, block_size, dtype=torch.float32, device=device)
    block_table = torch.zeros(num_seqs, max_blocks_per_seq, dtype=torch.int32, device=device)

    if mode == "per_tensor":
        # One scalar scale per whole tensor; the int8 is computed from it.
        k_int8, k_s = quantize_per_tensor(k)   # k_int8: [S, L, H, D]
        v_int8, v_s = quantize_per_tensor(v)
        k_scale_tok = None
        v_scale_tok = None
    else:
        # Per (seq, token, kv_head) scale over head_dim.
        k_int8, k_scale_tok = quantize_per_token(k, dim=-1)  # scale: [S, L, H]
        v_int8, v_scale_tok = quantize_per_token(v, dim=-1)

    next_block = 0
    for b in range(num_seqs):
        clen = int(context_lens[b])
        for j in range(blocks_per_seq[b]):
            phys = int(perm[next_block].item())
            next_block += 1
            block_table[b, j] = phys
            t0 = j * block_size
            for off in range(block_size):
                t = t0 + off
                if t >= clen:
                    break
                k_cache[phys, :, :, off, :] = k_int8[b, t].view(num_kv_heads, head_dim // x, x)
                v_cache[phys, :, :, off] = v_int8[b, t]
                if mode == "per_tensor":
                    k_scales[phys, :, off] = k_s
                    v_scales[phys, :, off] = v_s
                else:
                    k_scales[phys, :, off] = k_scale_tok[b, t]
                    v_scales[phys, :, off] = v_scale_tok[b, t]

    return k_cache, v_cache, k_scales, v_scales, block_table


def build_k_cache_int8_asym(k, context_lens, block_table, num_blocks,
                            block_size=16, x=8):
    """Asymmetric per-token INT8 K cache (K-only; pair with the symmetric builder
    for V). Reuses an existing `block_table` scatter so the physical layout
    matches the symmetric caches built alongside it. Returns
    (k_cache:int8, k_scales:fp32, k_zeros:fp32)."""
    num_seqs, _, num_kv_heads, head_dim = k.shape
    device = k.device
    k_int8, k_scale, k_zero = quantize_per_token_asym(k, dim=-1)  # [S,L,H] scale/zero

    k_cache = torch.zeros(num_blocks, num_kv_heads, head_dim // x, block_size, x,
                          dtype=torch.int8, device=device)
    k_scales = torch.zeros(num_blocks, num_kv_heads, block_size,
                           dtype=torch.float32, device=device)
    k_zeros = torch.zeros(num_blocks, num_kv_heads, block_size,
                          dtype=torch.float32, device=device)
    for b in range(num_seqs):
        for t in range(int(context_lens[b])):
            phys = int(block_table[b, t // block_size])
            off = t % block_size
            k_cache[phys, :, :, off, :] = k_int8[b, t].view(num_kv_heads, head_dim // x, x)
            k_scales[phys, :, off] = k_scale[b, t]
            k_zeros[phys, :, off] = k_zero[b, t]
    return k_cache, k_scales, k_zeros


def dequantize_kv(k_cache, v_cache, k_scales, v_scales, k_zeros=None):
    """Rebuild fp32 K/V caches from int8 + per-token scales, so the existing
    paged_decode_attention_reference can serve as the dequant oracle.

    k_cache: [num_blocks, num_kv_heads, head_dim/x, block_size, x] int8
    v_cache: [num_blocks, num_kv_heads, head_dim, block_size] int8
    scales:  [num_blocks, num_kv_heads, block_size] fp32 (per token-in-block)
    k_zeros: optional same-shape fp32 zero-points; asym dequant k = s·(q - z).
    """
    # K scale broadcasts over (head_dim/x, x): index [nb, H, 1, bs, 1].
    ks = k_scales[:, :, None, :, None]
    k_int = k_cache.to(torch.float32)
    if k_zeros is not None:
        k_int = k_int - k_zeros[:, :, None, :, None]
    k_deq = k_int * ks
    # V scale broadcasts over head_dim: index [nb, H, 1, bs].
    vs = v_scales[:, :, None, :]
    v_deq = v_cache.to(torch.float32) * vs
    return k_deq, v_deq
