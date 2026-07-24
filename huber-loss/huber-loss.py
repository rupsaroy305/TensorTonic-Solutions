import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    e=np.abs(np.asarray(y_true,float)-np.asarray(y_pred,float))
    return float(np.mean(np.where(e<=delta,0.5*e**2,delta*(e-0.5*delta))))