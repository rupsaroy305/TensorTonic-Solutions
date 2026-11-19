import numpy as np

def max_pool2d(x: np.ndarray, kernel_size: int, stride: int) -> np.ndarray:
    B,H,W,C=x.shape
    Hout=(H-kernel_size)//stride+1
    Wout=(W-kernel_size)//stride+1
    output=np.zeros((B,Hout,Wout,C),dtype=np.float64)
    for i in range(Hout):
        for j in range(Wout):
            patch=x[:,i*stride:i*stride+kernel_size,j*stride:j*stride+kernel_size,:]
            output[:,i,j,:]=np.max(patch,axis=(1,2))
    return output.astype(np.float64)