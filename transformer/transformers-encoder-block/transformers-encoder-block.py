import numpy as np

def layer_norm(x, gamma, beta, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads):
    B, T, d = Q.shape
    dk = d // num_heads
    Q, K, V = Q @ W_q, K @ W_k, V @ W_v

    def split(a):
        return a.reshape(B, -1, num_heads, dk).transpose(0, 2, 1, 3)

    Q, K, V = split(Q), split(K), split(V)
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dk)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return (weights @ V).transpose(0, 2, 1, 3).reshape(B, T, d) @ W_o

def feed_forward(x, W1, b1, W2, b2):
    return np.maximum(0, x @ W1 + b1) @ W2 + b2

def encoder_block(x, W_q, W_k, W_v, W_o, W1, b1, W2, b2,
                  gamma1, beta1, gamma2, beta2, num_heads):
    z = layer_norm(
        x + multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads),
        gamma1, beta1
    )
    return layer_norm(z + feed_forward(z, W1, b1, W2, b2), gamma2, beta2)