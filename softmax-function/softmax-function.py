import numpy as np

def softmax(x):
    x=np.asarray(x,dtype=float)
    if x.ndim==1:
        e=np.exp(x-np.max(x))
        return e/np.sum(e)
    e=np.exp(x-np.max(x,axis=1,keepdims=True))
    return e/np.sum(e,axis=1,keepdims=True)