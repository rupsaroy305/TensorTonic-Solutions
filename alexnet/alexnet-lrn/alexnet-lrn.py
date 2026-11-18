import numpy as np

def local_response_normalization(x: np.ndarray, k: float, n: int,
                                 alpha: float, beta: float) -> np.ndarray:
    B,H,W,C=x.shape
    r=n//2
    output=np.zeros_like(x,dtype=np.float64)
    for c in range(C):
        start=max(0,c-r)
        end=min(C,c+r+1)
        total=np.sum(x[:,:,:,start:end]**2,axis=3)
        output[:,:,:,c]=x[:,:,:,c]/(k+(alpha/n)*total)**beta
    return output.astype(np.float64)