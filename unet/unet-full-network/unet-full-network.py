import numpy as np

def unet_forward(x: np.ndarray, weights: dict) -> np.ndarray:
    def conv(x, kernel, bias):
        B, H, W, C = x.shape
        k = kernel.shape[0]
        p = k // 2
        xp = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))
        out = np.zeros((B, H, W, kernel.shape[-1]), dtype=np.float64)
        for i in range(k):
            for j in range(k):
                out += xp[:, i:i+H, j:j+W, :] @ kernel[i, j]
        return np.maximum(0, out + bias)
    skip = conv(x, weights["enc_kernel1"], weights["enc_bias1"])
    skip = conv(skip, weights["enc_kernel2"], weights["enc_bias2"])
    B, H, W, C = skip.shape
    x = skip.reshape(B, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))
    x = conv(x, weights["bridge_kernel1"], weights["bridge_bias1"])
    x = conv(x, weights["bridge_kernel2"], weights["bridge_bias2"])
    x = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
    x = np.maximum(0, x @ weights["W_up"] + weights["b_up"])
    H, W = x.shape[1:3]
    h, w = skip.shape[1:3]
    top, left = (h - H) // 2, (w - W) // 2
    skip = skip[:, top:top+H, left:left+W, :]
    x = np.concatenate((skip, x), axis=-1)
    x = conv(x, weights["dec_kernel1"], weights["dec_bias1"])
    x = conv(x, weights["dec_kernel2"], weights["dec_bias2"])
    return (x @ weights["W_out"] + weights["b_out"]).astype(np.float64)