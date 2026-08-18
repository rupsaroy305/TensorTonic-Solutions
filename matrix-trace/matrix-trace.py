import numpy as np

def matrix_trace(A):
    A=np.asarray(A)
    trace=0
    for i in range(A.shape[0]):
        trace+=A[i,i]
    return trace