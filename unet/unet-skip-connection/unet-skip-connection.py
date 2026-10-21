import numpy as np

def crop_and_concat(encoder_features: np.ndarray,
                    decoder_features: np.ndarray) -> np.ndarray:
    H, W = decoder_features.shape[1:3]
    h, w = encoder_features.shape[1:3]
    top = (h - H) // 2
    left = (w - W) // 2
    encoder_features = encoder_features[:, top:top+H, left:left+W, :]
    return np.concatenate((encoder_features, decoder_features), axis=-1).astype(np.float64)