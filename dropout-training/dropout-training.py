import numpy as np

def dropout(x, p=0.5, rng=None):
    x = np.asarray(x, dtype=float)

    if rng is None:
        rng = np.random

    keep = (rng.random(x.shape) >= p).astype(float)
    scale = 1.0 / (1.0 - p)

    dropout_pattern = keep * scale
    output = x * dropout_pattern

    return output, dropout_pattern