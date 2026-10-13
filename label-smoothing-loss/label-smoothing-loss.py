import math

def label_smoothing_loss(predictions: list, target: int, epsilon: float) -> float:
    k = len(predictions)
    loss = 0.0
    for i, p in enumerate(predictions):
        q = epsilon/k+(1-epsilon if i==target else 0)
        loss -= q*math.log(p)
    return loss