import numpy as np

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    B, T, d = Q.shape
    dk = d // num_heads
    Q, K, V = Q @ W_q, K @ W_k, V @ W_v
    Q = Q.reshape(B, T, num_heads, dk).transpose(0, 2, 1, 3)
    K = K.reshape(B, -1, num_heads, dk).transpose(0, 2, 1, 3)
    V = V.reshape(B, -1, num_heads, dk).transpose(0, 2, 1, 3)
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dk)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    out = weights @ V
    out = out.transpose(0, 2, 1, 3).reshape(B, T, d)
    return out @ W_o