import numpy as np

def swish(x):
    x=np.atleast_1d(np.asarray(x,float))
    s=np.where(x>=0,1/(1+np.exp(-x)),np.exp(x)/(1+np.exp(x)))
    return x*s