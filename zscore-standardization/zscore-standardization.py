import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    X=np.asarray(X,dtype=float)
    if X.ndim==1:
        m=np.mean(X)
        s=np.std(X)
    else:
        m=np.mean(X,axis=axis,keepdims=True)
        s=np.std(X,axis=axis,keepdims=True)
    return (X-m)/(s+eps)