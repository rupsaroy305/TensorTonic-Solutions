import numpy as np

def percentiles(x, q):
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)
    x_sorted = np.sort(x)
    n = len(x_sorted)
    positions = (q / 100) * (n - 1)
    lower = np.floor(positions).astype(int)
    upper = np.ceil(positions).astype(int)
    weight = positions - lower
    return x_sorted[lower] + weight * (x_sorted[upper] - x_sorted[lower])