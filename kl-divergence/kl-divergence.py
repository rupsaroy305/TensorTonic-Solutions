import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    mask = p > 0
    p = p[mask]
    q = np.maximum(q[mask], eps)
    return float(np.sum(p * np.log(p / q)))