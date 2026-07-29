import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> tuple:
    z=np.c_[np.atleast_2d(h_prev),np.atleast_2d(x_t)]
    i,c=sigmoid(z@W_i.T+b_i),np.tanh(z@W_c.T+b_c)
    return (i[0],c[0]) if np.ndim(h_prev)==1 else (i,c)