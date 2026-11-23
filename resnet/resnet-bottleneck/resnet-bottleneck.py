import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    x=np.array(x,dtype=np.float64)
    W1=np.array(W1,dtype=np.float64)
    W2=np.array(W2,dtype=np.float64)
    W3=np.array(W3,dtype=np.float64)
    y=x@W1
    y=np.maximum(y,0)
    y=y@W2
    y=np.maximum(y,0)
    y=y@W3
    shortcut=x if Ws is None else x@np.array(Ws,dtype=np.float64)
    y=np.maximum(y+shortcut,0)
    return np.round(y,4).tolist()