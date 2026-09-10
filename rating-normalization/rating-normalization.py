import numpy as np
def rating_normalization(matrix: list) -> list:
    matrix=np.array(matrix, dtype=float)
    result=np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        rated=matrix[i]!=0
        if np.any(rated):
            mean=np.mean(matrix[i][rated])
            result[i][rated]=matrix[i][rated]-mean
    return np.round(result, 6).tolist()