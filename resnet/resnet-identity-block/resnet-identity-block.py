import numpy as np

def identity_block(x, W1, W2):
    x=np.array(x,dtype=np.float64)
    W1=np.array(W1,dtype=np.float64)
    W2=np.array(W2,dtype=np.float64)
    y=x@W1.T
    y=np.maximum(y,0)
    y=y@W2.T
    y=np.maximum(y+x,0)
    return np.round(y,4).tolist()