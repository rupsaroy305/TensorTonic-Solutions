import numpy as np

def t_test_one_sample(x, mu0):
    x=np.asarray(x,dtype=float)
    n=len(x)
    if n<2:
        return 0.0
    m=np.mean(x)
    s=np.std(x,ddof=1)
    if s==0:
        return 0.0 if m==mu0 else float("inf") if m>mu0 else float("-inf")
    return float((m-mu0)/(s/np.sqrt(n)))