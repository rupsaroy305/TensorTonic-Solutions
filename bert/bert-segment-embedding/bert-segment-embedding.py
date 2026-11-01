import numpy as np

def bert_embeddings(token_ids: np.ndarray, segment_ids: np.ndarray,
                    token_embeddings: np.ndarray, position_embeddings: np.ndarray,
                    segment_embeddings: np.ndarray) -> np.ndarray:
    B, S = token_ids.shape
    token_vectors = token_embeddings[token_ids]
    position_vectors = position_embeddings[np.arange(S)]
    segment_vectors = segment_embeddings[segment_ids]
    return (token_vectors + position_vectors[None, :, :] + segment_vectors).astype(np.float64)