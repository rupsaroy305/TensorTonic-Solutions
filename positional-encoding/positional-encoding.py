import numpy as np

def positional_encoding(seq_len, d_model, base=10000):

    positions = np.arange(seq_len, dtype=float)[:, np.newaxis]
    dims = np.arange(0, d_model, 2, dtype=float)

    angles = positions / np.power(base, dims / d_model)

    pe = np.zeros((seq_len, d_model), dtype=float)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles[:, :d_model // 2])

    return pe