import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    try:
        x=np.asarray(matrix,float)
        if x.ndim!=2:return None
        if norm_type=='l1':n=np.sum(np.abs(x),axis=axis,keepdims=True)
        elif norm_type=='l2':n=np.sqrt(np.sum(x*x,axis=axis,keepdims=True))
        elif norm_type=='max':n=np.max(np.abs(x),axis=axis,keepdims=True)
        else:return None
        n=np.where(n==0,1,n)
        return x/n
    except:
        return None