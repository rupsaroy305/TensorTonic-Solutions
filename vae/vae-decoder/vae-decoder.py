import numpy as np

def vae_decoder(z: np.ndarray, W_dec: np.ndarray, b_dec: np.ndarray) -> np.ndarray:
    x=z@W_dec+b_dec
    return (1/(1+np.exp(-x))).astype(np.float64)