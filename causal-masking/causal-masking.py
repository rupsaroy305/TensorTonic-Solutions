import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    scores=np.array(scores, dtype=float, copy=True)
    T=scores.shape[-1]
    scores[..., np.triu(np.ones((T, T), dtype=bool), 1)]=mask_value
    return scores