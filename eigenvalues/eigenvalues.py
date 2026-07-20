import numpy as np

def calculate_eigenvalues(matrix):
    try:
        A=np.asarray(matrix,float)
        if A.ndim!=2 or A.shape[0]!=A.shape[1]:return None
        if A.size==0:return np.array([],dtype=complex)
        w=np.linalg.eigvals(A)
        return w[np.lexsort((w.imag,w.real))]
    except:
        return None