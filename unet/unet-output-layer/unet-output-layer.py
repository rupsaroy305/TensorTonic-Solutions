import numpy as np

def unet_output(features: np.ndarray, W_out: np.ndarray,
                b_out: np.ndarray) -> np.ndarray:
    return np.asarray(features@W_out+b_out,dtype=np.float64)