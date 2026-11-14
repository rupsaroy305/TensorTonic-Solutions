import numpy as np

def vgg_classifier(features: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                   W2: np.ndarray, b2: np.ndarray,
                   W3: np.ndarray, b3: np.ndarray) -> np.ndarray:
    x=features.reshape(features.shape[0],-1)
    x=x@W1+b1
    x=np.maximum(x,0)
    x=x@W2+b2
    x=np.maximum(x,0)
    logits=x@W3+b3
    return logits.astype(np.float64)