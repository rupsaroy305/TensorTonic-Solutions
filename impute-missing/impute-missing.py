import numpy as np

def impute_missing(X, strategy='mean'):
    X=np.array(X,dtype=float,copy=True)
    d=X.ndim==1
    if d:X=X[:,None]
    for i in range(X.shape[1]):
        c=X[:,i]
        v=0 if np.isnan(c).all() else np.nanmean(c) if strategy=="mean" else np.nanmedian(c)
        c[np.isnan(c)]=v
    return X.ravel() if d else X