import numpy as np

def kl_divergence(p, q, eps=1e-12):
    p=np.asarray(p,float)
    q=np.asarray(q,float)+eps
    return float(np.sum(np.where(p>0,p*np.log(p/q),0.0)))