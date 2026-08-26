import numpy as np

def classification_metrics(y_true: list[int], y_pred: list[int], average: str = "micro", pos_label: int = 1) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = np.mean(y_true == y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))

    tp = np.array([np.sum((y_true == c) & (y_pred == c)) for c in classes])
    fp = np.array([np.sum((y_true != c) & (y_pred == c)) for c in classes])
    fn = np.array([np.sum((y_true == c) & (y_pred != c)) for c in classes])

    p = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=tp + fp != 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=tp + fn != 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=p + r != 0)

    if average == "micro":
        P = tp.sum() / (tp.sum() + fp.sum())
        R = tp.sum() / (tp.sum() + fn.sum())
        F = 2 * P * R / (P + R) if P + R else 0.0
    elif average == "macro":
        P, R, F = p.mean(), r.mean(), f.mean()
    elif average == "weighted":
        w = np.array([np.sum(y_true == c) for c in classes])
        w = w / len(y_true)
        P, R, F = np.sum(p * w), np.sum(r * w), np.sum(f * w)
    else:
        i = np.where(classes == pos_label)[0][0]
        P, R, F = p[i], r[i], f[i]

    return {
        "accuracy": round(float(acc), 6),
        "precision": round(float(P), 6),
        "recall": round(float(R), 6),
        "f1": round(float(F), 6)
    }