import numpy as np

def hidden_update(h_prev: np.ndarray, h_tilde: np.ndarray,
                  z_t: np.ndarray) -> np.ndarray:

    h_prev,h_tilde,z_t=map(np.atleast_2d,(h_prev,h_tilde,z_t))
    h=z_t*h_prev+(1-z_t)*h_tilde
    return h[0] if h.shape[0]==1 else h