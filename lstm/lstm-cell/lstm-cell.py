import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> tuple:

    s=h_prev.ndim==1
    if s:h_prev,x_t,C_prev=h_prev[None],x_t[None],C_prev[None]
    z=np.c_[h_prev,x_t]
    f=sigmoid(z@W_f.T+b_f);i=sigmoid(z@W_i.T+b_i)
    C=f*C_prev+i*np.tanh(z@W_c.T+b_c)
    h=sigmoid(z@W_o.T+b_o)*np.tanh(C)
    return (h[0],C[0]) if s else (h,C)