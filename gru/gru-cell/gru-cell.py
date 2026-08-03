import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def gru_cell(x_t: np.ndarray, h_prev: np.ndarray,
             W_r: np.ndarray, W_z: np.ndarray, W_h: np.ndarray,
             b_r: np.ndarray, b_z: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    x_t,h_prev=map(np.atleast_2d,(x_t,h_prev))
    c=np.concatenate((h_prev,x_t),1)
    r=sigmoid(c@W_r.T+b_r)
    z=sigmoid(c@W_z.T+b_z)
    h=np.tanh(np.concatenate((r*h_prev,x_t),1)@W_h.T+b_h)
    h=z*h_prev+(1-z)*h
    return h[0] if h.shape[0]==1 else h