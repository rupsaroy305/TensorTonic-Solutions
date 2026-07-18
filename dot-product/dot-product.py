import numpy as np

def dot_product(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("Inputs must be 1D arrays of equal length")
    return float(np.dot(x, y))