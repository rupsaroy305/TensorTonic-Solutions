import numpy as np

def maxpool_forward(X, pool_size, stride):
    X=np.array(X)
    H,W=X.shape
    OH=(H-pool_size)//stride+1
    OW=(W-pool_size)//stride+1

    out=[]
    for i in range(OH):
        row=[]
        for j in range(OW):
            w=X[i*stride:i*stride+pool_size,
                j*stride:j*stride+pool_size]
            row.append(w.max().item())
        out.append(row)

    return out