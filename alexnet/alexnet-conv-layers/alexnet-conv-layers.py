import numpy as np

def alexnet_conv1(image: np.ndarray, kernel: np.ndarray, bias: np.ndarray,
                   stride: int, padding: int) -> np.ndarray:
    B,H,W,C=image.shape
    Kh,Kw,_,Cout=kernel.shape
    padded=np.pad(image,((0,0),(padding,padding),(padding,padding),(0,0)))
    Hout=(H+2*padding-Kh)//stride+1
    Wout=(W+2*padding-Kw)//stride+1
    output=np.zeros((B,Hout,Wout,Cout),dtype=np.float64)
    for i in range(Hout):
        for j in range(Wout):
            patch=padded[:,i*stride:i*stride+Kh,j*stride:j*stride+Kw,:]
            output[:,i,j,:]=np.tensordot(patch,kernel,axes=([1,2,3],[0,1,2]))+bias
    return output.astype(np.float64)