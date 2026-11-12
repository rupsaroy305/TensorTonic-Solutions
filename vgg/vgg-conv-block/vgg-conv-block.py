import numpy as np

def vgg_conv_block(x: np.ndarray, kernels: list,
                   biases: list) -> np.ndarray:
    for kernel,bias in zip(kernels, biases):
        kh, kw, in_channels, out_channels = kernel.shape
        padding= kh//2
        padded= np.pad(x,((0,0),(padding,padding), (padding, padding), (0, 0)),mode="constant")
        N, H, W, _ = x.shape
        output= np.zeros((N,H,W,out_channels),dtype=np.float64)
        for i in range(H):
            for j in range(W):
                region= padded[:,i:i+kh,j:j+kw,:]
                output[:,i,j,:]= np.tensordot(region,kernel,axes=([1,2,3],[0,1,2]))+bias
        x= np.maximum(output,0)
    return x.astype(np.float64)