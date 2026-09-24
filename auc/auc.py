import numpy as np

def auc(fpr: list, tpr: list) -> float:
    fpr=np.array(fpr,dtype=float)
    tpr=np.array(tpr,dtype=float)
    return float(np.sum((fpr[1:]-fpr[:-1])*(tpr[1:]+tpr[:-1])/2))