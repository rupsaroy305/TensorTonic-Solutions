import math

def log_loss(y_true, y_pred, eps=1e-15):
    import numpy as np
    p=np.clip(y_pred,eps,1-eps)
    return (-(np.array(y_true)*np.log(p)+(1-np.array(y_true))*np.log(1-p))).tolist()