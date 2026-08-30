import numpy as np

def detect_skew(train_dist: dict, serving_dist: dict, threshold: float = 0.2, eps: float = 1e-10) -> dict:
    result = {}

    for f in train_dist:
        t = np.array(train_dist[f], dtype=float)
        s = np.array(serving_dist[f], dtype=float)

        psi = np.sum((s - t) * np.log((s + eps) / (t + eps)))

        result[f] = {
            "psi": round(float(psi), 6),
            "skewed": bool(psi >= threshold)
        }

    return result