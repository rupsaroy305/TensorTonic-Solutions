import numpy as np
from math import erf
def gelu(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * x * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))
