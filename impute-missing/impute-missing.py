import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    X = np.array(X, dtype=float)
    result = X.copy()
    if X.ndim == 1:
        values = X[~np.isnan(X)]
        fill = 0.0 if len(values) == 0 else (
            np.mean(values) if strategy == "mean" else np.median(values)
        )
        result[np.isnan(result)] = fill
    else:
        for j in range(X.shape[1]):
            values = X[:, j][~np.isnan(X[:, j])]
            fill = 0.0 if len(values) == 0 else (
                np.mean(values) if strategy == "mean" else np.median(values)
            )
            result[np.isnan(result[:, j]), j] = fill
    return result