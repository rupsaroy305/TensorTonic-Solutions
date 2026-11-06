import numpy as np

def bert_fine_tuning_step(hidden_states: np.ndarray, labels: np.ndarray,
                          classifier_W: np.ndarray, classifier_b: np.ndarray,
                          learning_rate: float) -> dict:
    X = hidden_states[:, 0, :]
    logits = X @ classifier_W + classifier_b
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    B = len(labels)
    loss = -np.mean(np.log(probabilities[np.arange(B), labels]))
    dZ = probabilities.copy()
    dZ[np.arange(B), labels] -= 1
    dZ /= B
    grad_W = X.T @ dZ
    grad_b = np.sum(dZ, axis=0)
    new_classifier_W = classifier_W - learning_rate * grad_W
    new_classifier_b = classifier_b - learning_rate * grad_b
    return {"new_classifier_W": new_classifier_W.astype(np.float64),"new_classifier_b": new_classifier_b.astype(np.float64),"loss": float(loss)}