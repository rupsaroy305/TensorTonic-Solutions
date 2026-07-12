import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    X=np.asarray(X,dtype=float)
    if X.ndim==1:
        mn=np.min(X)
        mx=np.max(X)
    else:
        mn=np.min(X,axis=axis,keepdims=True)
        mx=np.max(X,axis=axis,keepdims=True)
    return (X-mn)/(mx-mn+eps)