import numpy as np
from math import exp,lgamma

def poisson_pmf_cdf(lam, k):
    f=lambda i:exp(-lam+i*np.log(lam)-lgamma(i+1))
    return f(k),sum(f(i) for i in range(k+1))