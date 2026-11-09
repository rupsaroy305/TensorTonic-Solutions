import torch
import torch.nn.functional as F

def dense_block(x: torch.Tensor, layers: list, growth_rate: int,
                eps: float = 1e-5) -> torch.Tensor:
    outputs= [x]
    for layer in layers:
        combined= torch.cat(outputs,dim=1)
        out= F.batch_norm(combined,layer["bn_mean"],layer["bn_var"],layer["bn_gamma"],layer["bn_beta"],training=False,eps=eps)
        out= F.relu(out)
        out= F.conv2d(out,layer["conv_weight"], padding=1)
        outputs.append(out)
    return torch.cat(outputs,dim=1).to(torch.float64)