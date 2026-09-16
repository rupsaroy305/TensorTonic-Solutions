import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred, dtype=float)
    correct_probs = y_pred[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(correct_probs)))