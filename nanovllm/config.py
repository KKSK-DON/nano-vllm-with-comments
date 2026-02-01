import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384 # 影响prefill 吞吐上限，太高容易OOM
    max_num_seqs: int = 512 # 影响并发上限 影响decode 吞吐
    max_model_len: int = 4096 # 决定graph 捕获时的block_tables 大小
    gpu_memory_utilization: float = 0.9 # 现存利用率，影响kv cache 块数， 太高容易OOM
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256 # 控制颗粒度，小更容易命中。影响缓存命中率和meta data
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
