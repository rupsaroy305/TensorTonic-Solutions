
import numpy as np
def mean_rating_imputation(ratings_matrix: list, mode: str) -> list:
    a = np.array(ratings_matrix, dtype=float)
    result = a.copy()

    if mode == "user":
        for i in range(a.shape[0]):
            mean = np.mean(a[i][a[i] != 0]) if np.any(a[i] != 0) else 0.0
            result[i][a[i] == 0] = mean
    else:
        for j in range(a.shape[1]):
            mean = np.mean(a[:, j][a[:, j] != 0]) if np.any(a[:, j] != 0) else 0.0
            result[:, j][a[:, j] == 0] = mean

    return result.tolist()