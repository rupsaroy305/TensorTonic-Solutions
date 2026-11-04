import numpy as np

def bert_pooler(hidden_states: np.ndarray, W_pool: np.ndarray,
                b_pool: np.ndarray) -> np.ndarray:
    cls_output=hidden_states[:,0,:]
    pooled=np.tanh(cls_output@ W_pool+b_pool)
    return pooled