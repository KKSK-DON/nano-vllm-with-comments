from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        # max_num_seqs and max_num_batched_tokens is for controlling every size of step
        # to avoid oom, computed by hbm
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # we may see diff design but waiting and running q is needed,
        # because there are two phase: prefill and decode
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        # There are many scheduling strategies. Here is prefill always first even decode exists.
        # vllm can run prefill and decode together
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            # below is because we do not need to re-compute the part of hitting prefix cache
            # prefix cache belongs to kvcache, no need extra hbm
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            # if there aren’t enough KV-cache blocks available to append a new token for this request, 
            # then it has to free up some HBM resources
            # because inference needs to allocate KV cache for the newly generated token
            # free up by preempting: kicking out running to waiting and deallocating its kvcache blocks
            while not self.block_manager.can_append(seq):
                # 1. preempt newest running seq 2. preempt current one
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else: # real inference process
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        """
        Notice there is a line assert scheduled in the code. 
        This is a key invariant: during the Decode phase, at least one request must be scheduled.
        Why is that guaranteed? Because of the self-preemption mechanism. 
        In the worst case, the current request frees its own blocks, 
        and then uses the freed blocks to append a new token for itself.
        As long as there is still at least one request in the running state, 
        the system will always be able to make progress.
        """
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        # reminder: put to the front not the end
        # to avoid it never getting scheduled
        self.waiting.appendleft(seq) 

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
