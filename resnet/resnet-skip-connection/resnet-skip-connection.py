import numpy as np

def compute_gradient_with_skip(gradients_F: list,x: np.ndarray)->np.ndarray:
    g=np.array(x,dtype=np.float64).copy()
    for J in gradients_F:
        J=np.array(J,dtype=np.float64)
        g=g@(np.eye(len(g))+J)
    return g.astype(np.float64)

def compute_gradient_without_skip(gradients_F: list,x: np.ndarray)->np.ndarray:
    g=np.array(x,dtype=np.float64).copy()
    for J in gradients_F:
        J=np.array(J,dtype=np.float64)
        g=g@J
    return g.astype(np.float64)