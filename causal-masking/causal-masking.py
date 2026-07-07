import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    scores=np.array(scores,dtype=float,copy=True)
    t=scores.shape[-1]
    mask=np.triu(np.ones((t,t),dtype=bool),1)
    scores[...,mask]=mask_value
    return scores