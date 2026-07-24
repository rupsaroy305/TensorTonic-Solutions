import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    a=np.atleast_2d(np.asarray(anchor,float))
    p=np.atleast_2d(np.asarray(positive,float))
    n=np.atleast_2d(np.asarray(negative,float))
    dp=np.sum((a-p)**2,axis=1)
    dn=np.sum((a-n)**2,axis=1)
    return float(np.maximum(0,dp-dn+margin).mean())