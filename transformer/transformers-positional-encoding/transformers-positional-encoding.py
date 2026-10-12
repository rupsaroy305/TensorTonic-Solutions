import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    pos = np.arange(seq_length)[:, None]
    i = np.arange(0, d_model, 2)
    angles = pos / (10000 ** (i / d_model))
    pe = np.zeros((seq_length, d_model), dtype=np.float64)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe