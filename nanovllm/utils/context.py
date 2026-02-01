from dataclasses import dataclass
import torch


@dataclass
class Context:
    """
    为什么用全局变量？因为 Context 需要被模型的各个层访问（主要是 Attention 层），
    而模型的 forward 签名是固定的 (input_ids, positions)。用全局变量可以在不改变模型接口的情况下传递额外信息。
    """
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None # kv 写入的物理槽位
    context_lens: torch.Tensor | None = None # 每个序列的上下文长度
    block_tables: torch.Tensor | None = None # 物理块表

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
