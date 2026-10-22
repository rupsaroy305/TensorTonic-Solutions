import numpy as np

def unet_bottleneck(x: np.ndarray, kernel1: np.ndarray, bias1: np.ndarray,
                    kernel2: np.ndarray, bias2: np.ndarray) -> np.ndarray:
    def conv(x, kernel, bias):
        B, H, W, C = x.shape
        k = kernel.shape[0]
        p = k // 2
        x_pad = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))
        out = np.zeros((B, H, W, kernel.shape[-1]), dtype=np.float64)
        for i in range(k):
            for j in range(k):
                out += x_pad[:, i:i+H, j:j+W, :] @ kernel[i, j]
        return np.maximum(0, out + bias)
    x = conv(x, kernel1, bias1)
    return conv(x, kernel2, bias2).astype(np.float64)