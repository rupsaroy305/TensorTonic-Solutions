import numpy as np

def candidate_hidden(h_prev: np.ndarray, x_t: np.ndarray, r_t: np.ndarray,
                     W_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    h_prev,x_t,r_t=map(np.atleast_2d,(h_prev,x_t,r_t))
    h=np.tanh(np.concatenate((r_t*h_prev,x_t),1)@W_h.T+b_h)
    return h[0] if h.shape[0]==1 else h