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
