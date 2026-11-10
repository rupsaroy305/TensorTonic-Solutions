import torch
import torch.nn.functional as F

def transition_layer(x: torch.Tensor, bn_gamma: torch.Tensor, bn_beta: torch.Tensor,
                     bn_mean: torch.Tensor, bn_var: torch.Tensor,
                     conv_weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x= F.batch_norm(x,bn_mean,bn_var,bn_gamma,bn_beta,training=False,eps=eps)
    x= F.relu(x)
    x= F.conv2d(x, conv_weight, bias=None)
    x= F.avg_pool2d(x, kernel_size=2, stride=2)
    return x.to(torch.float64)