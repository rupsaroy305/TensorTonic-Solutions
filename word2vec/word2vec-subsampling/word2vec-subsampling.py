import torch

def subsample_keep_probs(counts: torch.Tensor,
                         t: float = 1e-5) -> torch.Tensor:
    f=counts/counts.sum()
    return torch.minimum(torch.ones_like(f),torch.sqrt(t/f))