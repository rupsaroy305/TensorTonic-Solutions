import numpy as np
def binary_focal_loss(predictions, targets, alpha, gamma):
    p=np.array(predictions,float)
    t=np.array(targets)
    pt=np.where(t==1,p,1-p)
    return float(np.mean(-alpha*(1-pt)**gamma*np.log(pt)))