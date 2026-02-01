import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1)) 
        probs = torch.softmax(logits, dim=-1)
        # https://huggingface.co/blog/cxdu/fastsampling => argmax(log(p_i) + G_i)
        # G_i​=−logE_i​∼Gumbel(0,1)
        # logp_i​+G_i​=logp_i​−logE_i​=log(​p_i/E_i​​)
        # argmax ​log (p_i/E_i​​)​=argmax ​(p_i/E_i​​)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        # E = torch.empty_like(probs).exponential_(1)
        # E = E.clamp_min_(1e-10) to avoid 0 division
        # sample_tokens = (probs / E).argmax(dim=-1)
        return sample_tokens
