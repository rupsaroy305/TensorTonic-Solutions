import numpy as np

def vae_loss(x: np.ndarray, reconstruction: np.ndarray,
             mu: np.ndarray, log_var: np.ndarray) -> dict:
    reconstruction_loss=np.mean(np.sum((x-reconstruction)**2,axis=1))
    kl_loss=-0.5*np.mean(np.sum(1 + log_var - mu**2 - np.exp(log_var),axis=1))
    total_loss=reconstruction_loss+kl_loss
    return {"total_loss": float(total_loss),"reconstruction_loss": float(reconstruction_loss),"kl_loss": float(kl_loss)}