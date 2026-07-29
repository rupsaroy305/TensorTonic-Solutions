import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def output_gate(h_prev: np.ndarray, x_t: np.ndarray, C_t: np.ndarray,
                W_o: np.ndarray, b_o: np.ndarray) -> tuple:
    z=np.c_[np.atleast_2d(h_prev),np.atleast_2d(x_t)]
    o=sigmoid(z@W_o.T+b_o); h=o*np.tanh(np.atleast_2d(C_t))
    return (o[0],h[0]) if np.ndim(h_prev)==1 else (o,h)