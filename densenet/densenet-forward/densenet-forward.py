import torch
import torch.nn.functional as F

def densenet_forward(x: torch.Tensor, weights: dict, growth_rate: int,
                     eps: float = 1e-5) -> torch.Tensor:
    x = F.conv2d(x, weights["stem_conv"], padding=1)
    for i, block in enumerate(weights["blocks"]):
        outputs = [x]
        for layer in block:
            combined = torch.cat(outputs, dim=1)
            out = F.batch_norm(combined,layer["bn_mean"],layer["bn_var"],layer["bn_gamma"],layer["bn_beta"],training=False,eps=eps)
            out = F.relu(out)
            out = F.conv2d(out, layer["conv_weight"], padding=1)
            outputs.append(out)
        x = torch.cat(outputs, dim=1)
        if i < len(weights["transitions"]):
            transition = weights["transitions"][i]

            x = F.batch_norm(x,transition["bn_mean"],transition["bn_var"],transition["bn_gamma"],transition["bn_beta"],training=False,eps=eps)

            x = F.relu(x)
            x = F.conv2d(x, transition["conv_weight"], bias=None)
            x = F.avg_pool2d(x, kernel_size=2, stride=2)

    x = F.batch_norm(x,weights["final_bn_mean"],weights["final_bn_var"],weights["final_bn_gamma"],weights["final_bn_beta"],training=False,eps=eps)
    x = F.relu(x)
    x = torch.mean(x, dim=(2, 3))
    logits = x @ weights["fc_weight"].T + weights["fc_bias"]
    return logits.to(torch.float64)