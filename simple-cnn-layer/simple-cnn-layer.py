import numpy as np

def conv2d(x, W, b):
    x=np.asarray(x,float)
    W=np.asarray(W,float)
    b=np.asarray(b,float)
    N,C,H,W0=x.shape
    F,_,KH,KW=W.shape
    OH=H-KH+1
    OW=W0-KW+1
    out=np.zeros((N,F,OH,OW))
    for n in range(N):
        for f in range(F):
            for i in range(OH):
                for j in range(OW):
                    out[n,f,i,j]=np.sum(x[n,:,i:i+KH,j:j+KW]*W[f])+b[f]
    return out