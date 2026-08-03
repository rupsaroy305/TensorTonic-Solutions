import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def update_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_z: np.ndarray, b_z: np.ndarray) -> np.ndarray:
    h_prev=np.atleast_2d(h_prev)
    x_t=np.atleast_2d(x_t)
    z=sigmoid(np.concatenate((h_prev,x_t),1)@W_z.T+b_z)
    return z[0] if z.shape[0]==1 else z