import numpy as np

def cross_entropy_loss(y_true,y_pred):
    y_true=np.array(y_true)
    y_pred=np.array(y_pred,dtype=float)
    loss=-np.log(y_pred[np.arange(len(y_true)),y_true])
    return float(np.mean(loss))