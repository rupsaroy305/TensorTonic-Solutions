import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def reset_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_r: np.ndarray, b_r: np.ndarray) -> np.ndarray:
    c=np.concatenate((np.atleast_2d(h_prev),np.atleast_2d(x_t)),1)@W_r.T+b_r
    c=sigmoid(c)
    return c[0] if np.ndim(h_prev)==1 else c