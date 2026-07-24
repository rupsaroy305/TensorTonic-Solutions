import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    Z1=np.asarray(Z1,float)
    Z2=np.asarray(Z2,float)
    S=Z1@Z2.T/temperature
    S-=S.max(axis=1,keepdims=True)
    expS=np.exp(S)
    p=np.diag(expS)/expS.sum(axis=1)
    return float(-np.log(p).mean())