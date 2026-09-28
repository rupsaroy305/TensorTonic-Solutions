import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    n=len(token_ids)
    pairs=[]
    for i in range(n):
        for j in range(max(0,i-window),min(n,i+window+1)):
            if i!=j:
                pairs.append([token_ids[i].item(),token_ids[j].item()])
    return torch.tensor(pairs, dtype=torch.int64).reshape(-1,2)