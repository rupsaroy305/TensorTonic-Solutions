import numpy as np

def mean_average_precision(y_true_list: list, y_score_list: list, k: int | None = None) -> dict:
    ap_per_query = []
    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        order = np.argsort(-y_score)
        y_true = y_true[order]
        total_relevant = np.sum(y_true)
        if total_relevant == 0:
            ap_per_query.append(0.0)
            continue
        if k is None:
            limit = len(y_true)
        else:
            limit = min(k, len(y_true))
        hits = 0
        ap = 0.0
        for i in range(limit):
            if y_true[i] == 1:
                hits += 1
                ap += hits / (i + 1)
        ap /= total_relevant
        ap_per_query.append(round(float(ap), 6))
    map_value = np.mean(ap_per_query) if ap_per_query else 0.0
    return {
        "map_value": round(float(map_value), 6),
        "ap_per_query": ap_per_query
    }