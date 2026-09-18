import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    X = np.array(X, dtype=float)

    cov = np.cov(X, rowvar=False)
    std = np.std(X, axis=0, ddof=1)

    corr = cov / np.outer(std, std)
    corr[:, std == 0] = np.nan
    corr[std == 0, :] = np.nan

    return corr