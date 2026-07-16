import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    y_true=np.asarray(y_true)
    y_score=np.asarray(y_score,dtype=float)
    if y_true.shape!=y_score.shape or np.any((y_true!=1)&(y_true!=-1)):
        raise ValueError()
    loss=np.maximum(0,margin-y_true*y_score)
    if reduction=="mean": return float(loss.mean())
    if reduction=="sum": return float(loss.sum())
    raise ValueError()