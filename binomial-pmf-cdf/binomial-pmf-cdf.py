import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    i=np.arange(k+1)
    return float(comb(n,k)*p**k*(1-p)**(n-k)),float(np.sum(comb(n,i)*p**i*(1-p)**(n-i)))