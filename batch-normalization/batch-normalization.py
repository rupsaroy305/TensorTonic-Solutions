import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    x=np.asarray(x,dtype=float)
    gamma=np.asarray(gamma,dtype=float)
    beta=np.asarray(beta,dtype=float)
    if x.ndim==2:
        m=x.mean(0);v=((x-m)**2).mean(0)
        return gamma*(x-m)/np.sqrt(v+eps)+beta
    if x.ndim==4:
        m=x.mean((0,2,3),keepdims=True);v=((x-m)**2).mean((0,2,3),keepdims=True)
        return gamma.reshape(1,-1,1,1)*(x-m)/np.sqrt(v+eps)+beta.reshape(1,-1,1,1)
    raise ValueError()