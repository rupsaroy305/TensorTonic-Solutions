import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);y=np.asarray(y,dtype=float)
    if np.any((y!=0)&(y!=1)): raise ValueError()
    d=np.linalg.norm(a-b,axis=-1)
    loss=y*d**2+(1-y)*np.maximum(0,margin-d)**2
    if reduction=="mean": return float(np.mean(loss))
    if reduction=="sum": return float(np.sum(loss))
    raise ValueError()