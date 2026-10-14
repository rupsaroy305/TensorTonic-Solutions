import torch
import math
import torch.nn.functional as F

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    scores = Q @ K.transpose(-2, -1) / math.sqrt(K.shape[-1])
    return F.softmax(scores, dim=-1) @ V