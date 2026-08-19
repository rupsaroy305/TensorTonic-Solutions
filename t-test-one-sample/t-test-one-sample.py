import numpy as np
def t_test_one_sample(x, mu0):
    x=np.asarray(x,dtype=float)
    n=len(x)
    mean=np.mean(x)
    s=np.sqrt(np.sum((x-mean)**2)/(n-1))
    se=s/np.sqrt(n)
    t_stat=(mean-mu0)/se
    return float(t_stat)