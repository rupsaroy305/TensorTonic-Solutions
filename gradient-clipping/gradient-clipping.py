import numpy as np

def clip_gradients(g, max_norm):
    g=np.asarray(g,dtype=float)
    if max_norm<=0:return g
    norm=np.linalg.norm(g)
    if norm==0 or norm<=max_norm:return g
    return g*(max_norm/norm)