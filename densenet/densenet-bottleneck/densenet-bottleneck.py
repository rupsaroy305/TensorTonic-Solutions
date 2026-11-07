import torch
import torch.nn.functional as F

def bottleneck_layer(x: torch.Tensor, bn1_gamma: torch.Tensor, bn1_beta: torch.Tensor,
                     bn1_mean: torch.Tensor, bn1_var: torch.Tensor,
                     conv1_weight: torch.Tensor, bn2_gamma: torch.Tensor,
                     bn2_beta: torch.Tensor, bn2_mean: torch.Tensor,
                     bn2_var: torch.Tensor, conv2_weight: torch.Tensor,
                     eps: float = 1e-5) -> torch.Tensor:
    x= F.batch_norm(x, bn1_mean, bn1_var, bn1_gamma, bn1_beta, training=False, eps=eps)
    x= F.relu(x)
    x= F.conv2d(x, conv1_weight, bias=None, padding=0)
    x= F.batch_norm(x, bn2_mean, bn2_var,bn2_gamma, bn2_beta,training=False, eps=eps)
    x= F.relu(x)
    y= F.conv2d(x, conv2_weight, bias=None, padding=1)
    return y.to(torch.float64)