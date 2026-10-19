import numpy as np

def unet_encoder_block(x: np.ndarray, kernel1: np.ndarray, bias1: np.ndarray,
                       kernel2: np.ndarray, bias2: np.ndarray) -> dict:
    def conv(x, k, b):
        B, H, W, C = x.shape
        kh, kw, _, out_c = k.shape
        ph, pw = kh // 2, kw // 2
        xp = np.pad(x, ((0,0), (ph,ph), (pw,pw), (0,0)))
        y = np.zeros((B, H, W, out_c))
        for i in range(kh):
            for j in range(kw):
                y += xp[:, i:i+H, j:j+W, :] @ k[i, j]
        return np.maximum(0, y + b)
    skip = conv(conv(x, kernel1, bias1), kernel2, bias2)
    pooled = skip.reshape(
        skip.shape[0], skip.shape[1]//2, 2,
        skip.shape[2]//2, 2, skip.shape[3]
    ).max(axis=(2, 4))
    return {"pooled": pooled, "skip": skip}