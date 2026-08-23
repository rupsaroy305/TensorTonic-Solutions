import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma)
    beta = np.asarray(beta)

    if x.ndim == 2:
        mean = np.mean(x, axis=0)
        var = np.mean((x - mean) ** 2, axis=0)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
    var = np.mean((x - mean) ** 2, axis=(0, 2, 3), keepdims=True)

    return gamma.reshape(1, -1, 1, 1) * (x - mean) / np.sqrt(var + eps) + beta.reshape(1, -1, 1, 1)