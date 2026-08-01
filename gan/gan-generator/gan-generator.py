import numpy as np

def generator(z, W, b):
    return np.tanh(np.array(z) @ np.array(W) + np.array(b))