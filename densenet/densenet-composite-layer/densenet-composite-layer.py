import torch
import torch.nn.functional as F

def composite_layer(x: torch.Tensor, bn_gamma: torch.Tensor, bn_beta: torch.Tensor,
                    bn_mean: torch.Tensor, bn_var: torch.Tensor,
                    conv_weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = (x - bn_mean[None, :, None, None]) / torch.sqrt(bn_var[None, :, None, None] + eps)
    x = x * bn_gamma[None, :, None, None] + bn_beta[None, :, None, None]
    x = F.relu(x)
    return F.conv2d(x, conv_weight, stride=1, padding=1)