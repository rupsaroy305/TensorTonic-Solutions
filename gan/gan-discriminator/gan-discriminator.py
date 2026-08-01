import numpy as np

def discriminator(x, W):
    return 1/(1+np.exp(-(np.array(x)@np.array(W))))