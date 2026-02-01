import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """
    把当前计算的 KV 写入缓存，然后执行注意力计算。
    难点在于如何与分页的 KV Cache 高效配合。
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        1. prepare_prefill 检测到 cu_seqlens_k[-1] > cu_seqlens_q[-1]
            → 说明部分 token 已缓存
            → 构造 block_tables

        2. Attention.forward:
            a. store_kvcache 写入未缓存部分的 KV(slot_mapping 只包含未缓存的槽位）
            b. block_tables 不为 None, 所以把 k, v 替换为 k_cache, v_cache
            c. flash_attn_varlen_func 通过 block_table 读取分页的完整 KV
        e.g.
            cu_seqlens_q = [0, 44]     # Query 只有 44 个 token(300 - 256)
            cu_seqlens_k = [0, 300]    # Key 有 300 个 token
            slot_mapping = [3072, 3073, ..., 3115]  # 44 个槽位, 块 12 的位置 0-43
            block_tables = [[5, 12]]   # 两个块：块 5(已缓存), 块 12(新写入)
            Attention 执行时：
                Q 来自当前计算的 44 个 token
                K/V 来自缓存，通过 block_table 索引读取全部 300 个 token 的 KV
        
        KV Cache 的内存布局
            每层的 KV Cache 形状是：
                k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
                v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            物理槽位的计算：
                slot = block_id * block_size + offset_in_block
            在缓存中的位置：
                k_cache[block_id, offset_in_block, :, :]  # 一个 token 的所有 KV head
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            """
            当 block_table 不为 None 时, Flash Attention 会从分页的 KV Cache 中读取 K/V,
            而不是用传入的 k/v 参数。这就是 Prefix Cache 的实现方式。
            """
            o = flash_attn_varlen_func(
                q, # [total_q, num_heads, head_dim]
                k, # [total_k, num_kv_heads, head_dim]
                v, # [total_k, num_kv_heads, head_dim]
                max_seqlen_q=context.max_seqlen_q, # int
                cu_seqlens_q=context.cu_seqlens_q, # [batch_size + 1]
                max_seqlen_k=context.max_seqlen_k, # int
                cu_seqlens_k=context.cu_seqlens_k, # [batch_size + 1]
                softmax_scale=self.scale, 
                causal=True, 
                block_table=context.block_tables # [batch_size, max_blocks] or None
            )
        else:    # decode
            o = flash_attn_with_kvcache(
                q.unsqueeze(1), # [batch_size, 1, num_heads, head_dim]
                k_cache, # [num_blocks, block_size, num_kv_heads, head_dim]
                v_cache, # [num_blocks, block_size, num_kv_heads, head_dim]
                cache_seqlens=context.context_lens, # [batch_size] avoid to read padding
                block_table=context.block_tables, # [batch_size, max_blocks]
                softmax_scale=self.scale, 
                causal=True
            )
        return o
