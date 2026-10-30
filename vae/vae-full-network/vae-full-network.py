import numpy as np

def vae_forward(x: np.ndarray, epsilon: np.ndarray,
                W_mu: np.ndarray, b_mu: np.ndarray,
                W_logvar: np.ndarray, b_logvar: np.ndarray,
                W_dec: np.ndarray, b_dec: np.ndarray) -> dict:
    mu= x@W_mu+b_mu
    log_var= x@W_logvar+b_logvar
    z= mu+np.exp(0.5*log_var)*epsilon
    logits= z@W_dec+b_dec
    reconstruction= 1/(1+np.exp(-logits))
    return {"reconstruction": reconstruction.astype(np.float64),"mu": mu.astype(np.float64),"log_var": log_var.astype(np.float64),"z": z.astype(np.float64)}