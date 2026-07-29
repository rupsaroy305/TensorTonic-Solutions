import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

#def forget_gate(h_prev: np.ndarray, x_t: np.ndarray,
 #               W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:

def forget_gate(h_prev, x_t, W_f, b_f):
    if h_prev.ndim == 1:
        h_prev = h_prev[None, :]
        x_t = x_t[None, :]
        return sigmoid(np.c_[h_prev, x_t] @ W_f.T + b_f)[0]
    return sigmoid(np.c_[h_prev, x_t] @ W_f.T + b_f)