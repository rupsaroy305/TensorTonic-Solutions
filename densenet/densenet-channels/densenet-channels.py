import math
import torch

def densenet_channel_counts(stem_channels: int, growth_rate: int,
                            block_layers: list, compression: float) -> torch.Tensor:
    channels = [stem_channels]
    C = stem_channels
    for i, n in enumerate(block_layers):
        C = C + n * growth_rate
        channels.append(C)
        if i < len(block_layers) - 1:
            C = math.floor(compression * C)
            channels.append(C)
    return torch.tensor(channels, dtype=torch.int64)