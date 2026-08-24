import numpy as np

def pca_projection(X: list, k: int) -> list:
    X = np.asarray(X, dtype=float)
    Xc = X - np.mean(X, axis=0)
    C = (Xc.T @ Xc) / (X.shape[0] - 1)
    values, vectors = np.linalg.eigh(C)
    idx = np.argsort(values)[::-1][:k]
    W = vectors[:, idx]
    return (Xc @ W).tolist()