import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor, pos_vec: torch.Tensor,
              neg_vecs: torch.Tensor) -> torch.Tensor:
    pos=torch.nn.functional.softplus(-torch.dot(center_vec, pos_vec))
    neg=torch.nn.functional.softplus(neg_vecs @ center_vec).sum()
    return pos+neg