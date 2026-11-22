import numpy as np

def vgg_features(x: np.ndarray, config: list,
                 kernels: list, biases: list) -> np.ndarray:
    x=np.array(x,dtype=np.float64)
    kernel_idx=0
    for layer in config:
        if layer=="M":
            B,H,W,C=x.shape
            x=x.reshape(B,H//2,2,W//2,2,C).max(axis=(2,4))
        else:
            kernel=np.array(kernels[kernel_idx],dtype=np.float64)
            bias=np.array(biases[kernel_idx],dtype=np.float64)
            kernel_idx+=1
            kh,kw,_,out_channels=kernel.shape
            padding_h=kh//2
            padding_w=kw//2
            padded=np.pad(x,((0,0),(padding_h,padding_h),(padding_w,padding_w),(0,0)))
            B,H,W,C=x.shape
            output=np.zeros((B,H,W,out_channels),dtype=np.float64)
            for i in range(H):
                for j in range(W):
                    patch=padded[:,i:i+kh,j:j+kw,:]
                    output[:,i,j,:]=np.tensordot(patch,kernel,axes=([1,2,3],[0,1,2]))+bias
            x=np.maximum(output,0)
    return x.astype(np.float64)