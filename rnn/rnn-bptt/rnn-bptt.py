import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,
                     x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    dt = (1 - h_t**2) * dh_next
    return dt @ W_hh, dt.T @ h_prev