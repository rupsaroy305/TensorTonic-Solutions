import numpy as np
def max_pooling_2d(X: list, pool_size: int) -> list:
    X=np.array(X)
    h,w=X.shape
    out=[]
    for i in range(0,h-pool_size+1,pool_size):
        row=[]
        for j in range(0,w-pool_size+1,pool_size):
            row.append(np.max(X[i:i+pool_size,j:j+pool_size]).item())
        out.append(row)
    return out