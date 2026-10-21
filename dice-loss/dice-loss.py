import numpy as np

def dice_loss(p: list, y: list, eps: float = 1e-8) -> float:
    p = np.array(p)
    y = np.array(y)
    dice = (2 * np.sum(p * y) + eps) / (np.sum(p) + np.sum(y) + eps)
    return float(1 - dice)