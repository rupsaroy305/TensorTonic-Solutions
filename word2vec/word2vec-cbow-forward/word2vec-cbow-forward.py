import torch
import torch.nn.functional as F

def cbow_forward(context_ids: torch.Tensor, target_id: int,
                 W_in: torch.Tensor, W_out: torch.Tensor) -> torch.Tensor:
    h=W_in[context_ids].mean(dim=0)
    z=W_out@h
    return F.cross_entropy(z.unsqueeze(0),torch.tensor([target_id]))