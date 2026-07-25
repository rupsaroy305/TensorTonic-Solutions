import numpy as np

def dice_loss(p, y, eps=1e-8):
    p=np.asarray(p,dtype=float)
    y=np.asarray(y,dtype=float)
    d=(2*np.sum(p*y)+eps)/(np.sum(p)+np.sum(y)+eps)
    return float(1-d)