import numpy as np

def focal_loss(p, y, gamma=2.0):
    p=np.asarray(p,dtype=float)
    y=np.asarray(y,dtype=float)
    p=np.clip(p,1e-15,1-1e-15)
    loss=-(1-p)**gamma*y*np.log(p)-p**gamma*(1-y)*np.log(1-p)
    return float(loss.mean())