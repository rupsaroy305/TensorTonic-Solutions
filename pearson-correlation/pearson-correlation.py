import numpy as np

def pearson_correlation(X):
    try:
        X=np.asarray(X,float)
        if X.ndim!=2 or X.shape[0]<2:return None
        X=X-X.mean(0)
        C=X.T@X/(X.shape[0]-1)
        s=np.sqrt(np.diag(C))
        R=C/np.outer(s,s)
        z=s==0
        if z.any():
            R[z,:]=R[:,z]=np.nan
        return R
    except:
        return None