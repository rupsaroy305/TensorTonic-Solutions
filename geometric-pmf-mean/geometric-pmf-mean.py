import numpy as np

def geometric_pmf_mean(k, p):
    k=np.asarray(k)
    return (1-p)**(k-1)*p,float(1/p)