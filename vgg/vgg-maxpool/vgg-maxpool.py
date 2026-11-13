import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    B, H, W, C = x.shape
    result=np.zeros((B,H//2,W//2,C),dtype=np.float64)
    for b in range(B):
        for i in range(H//2):
            for j in range(W//2):
                for c in range(C):
                    result[b,i,j,c]=max(
                        x[b,2*i,2*j,c],
                        x[b,2*i,2*j+1,c],
                        x[b,2*i+1,2*j,c],
                        x[b,2*i+1,2*j+1,c]
                    )
    return result