import numpy as np

def unet_decoder_block(x: np.ndarray, skip: np.ndarray,
                       W_up: np.ndarray, b_up: np.ndarray,
                       kernel1: np.ndarray, bias1: np.ndarray,
                       kernel2: np.ndarray, bias2: np.ndarray) -> np.ndarray:
    x = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
    x = np.maximum(0, x @ W_up + b_up)
    H, W = x.shape[1:3]
    sh, sw = skip.shape[1:3]
    h_start = (sh - H) // 2
    w_start = (sw - W) // 2
    skip = skip[:, h_start:h_start + H, w_start:w_start + W, :]
    x = np.concatenate([skip, x], axis=-1)
    
    def conv_relu(x, kernel, bias):
        B, H, W, _ = x.shape
        kh, kw, _, out_c = kernel.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(
            x,
            ((0, 0), (ph, ph), (pw, pw), (0, 0)),
            mode="constant"
        )
        out = np.zeros((B, H, W, out_c), dtype=np.float64)
        for i in range(kh):
            for j in range(kw):
                out += padded[:, i:i + H, j:j + W, :] @ kernel[i, j]
        return np.maximum(0, out + bias)
    x = conv_relu(x, kernel1, bias1)
    x = conv_relu(x, kernel2, bias2)
    return x.astype(np.float64)