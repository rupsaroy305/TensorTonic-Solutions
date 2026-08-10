import numpy as np

def ndcg(relevance_scores, k):
    scores = np.asarray(relevance_scores, dtype=float)
    k = min(k, len(scores))

    positions = np.arange(1, k + 1)
    discounts = np.log2(positions + 1)

    # DCG of the given ranking
    dcg = np.sum((2 ** scores[:k] - 1) / discounts)

    # DCG of the ideal ranking
    ideal_scores = np.sort(scores)[::-1][:k]
    idcg = np.sum((2 ** ideal_scores - 1) / discounts)

    if idcg == 0:
        return 0.0

    return float(dcg / idcg)