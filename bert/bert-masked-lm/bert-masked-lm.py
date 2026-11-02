import numpy as np

def apply_mlm_mask(token_ids: np.ndarray, mask_positions: np.ndarray,
                   replace_probs: np.ndarray, random_tokens: np.ndarray,
                   mask_token_id: int = 103) -> dict:
    masked_ids= token_ids.copy()
    labels= np.full(token_ids.shape,-100,dtype=np.int64)
    labels[mask_positions]= token_ids[mask_positions]
    mask= mask_positions & (replace_probs<0.8)
    random= mask_positions & (replace_probs>=0.8) & (replace_probs<0.9)
    masked_ids[mask]= mask_token_id
    masked_ids[random]= random_tokens[random]
    return {"masked_ids": masked_ids.astype(np.int64),"labels": labels}