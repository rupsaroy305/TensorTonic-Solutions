import numpy as np

def matrix_inverse(A):
    A=np.array(A,dtype=float)
    if A.ndim!=2 or A.shape[0]!=A.shape[1]:
        return None
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None