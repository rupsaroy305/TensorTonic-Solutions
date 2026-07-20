import numpy as np

def covariance_matrix(X):
    try:
        X=np.asarray(X,float)
        if X.ndim!=2 or X.shape[0]<2:return None
        X-=X.mean(0)
        return X.T@X/(X.shape[0]-1)
    except:
        return None