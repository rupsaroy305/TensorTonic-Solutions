import numpy as np

def batch_norm_block(x,W1,W2,gamma1,beta1,gamma2,beta2,mode):
    x=np.array(x,dtype=np.float64)
    W1=np.array(W1,dtype=np.float64)
    W2=np.array(W2,dtype=np.float64)
    gamma1=np.array(gamma1,dtype=np.float64)
    beta1=np.array(beta1,dtype=np.float64)
    gamma2=np.array(gamma2,dtype=np.float64)
    beta2=np.array(beta2,dtype=np.float64)
    if mode=="post":
        y=x@W1
        mean=np.mean(y,axis=0)
        var=np.var(y,axis=0)
        y=gamma1*(y-mean)/np.sqrt(var+1e-5)+beta1
        y=np.maximum(y,0)
        y=y@W2
        mean=np.mean(y,axis=0)
        var=np.var(y,axis=0)
        y=gamma2*(y-mean)/np.sqrt(var+1e-5)+beta2
        y=np.maximum(y+x,0)
    else:
        mean=np.mean(x,axis=0)
        var=np.var(x,axis=0)
        y=gamma1*(x-mean)/np.sqrt(var+1e-5)+beta1
        y=np.maximum(y,0)
        y=y@W1
        mean=np.mean(y,axis=0)
        var=np.var(y,axis=0)
        y=gamma2*(y-mean)/np.sqrt(var+1e-5)+beta2
        y=np.maximum(y,0)
        y=y@W2
        y=y+x
    return {
        "output":np.round(y,4).tolist(),
        "mode":mode
    }