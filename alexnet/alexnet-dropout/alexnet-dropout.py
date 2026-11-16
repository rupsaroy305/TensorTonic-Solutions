import numpy as np

def dropout(x: np.ndarray, p: float, training: bool,
            mask: np.ndarray) -> np.ndarray:
    if training:
        return (x*mask/(1-p)).astype(np.float64)
    return x.copy().astype(np.float64)