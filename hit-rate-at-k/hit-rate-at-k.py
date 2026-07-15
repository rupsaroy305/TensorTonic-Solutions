def hit_rate_at_k(recommendations, ground_truth, k):
    if len(recommendations)==0:
        return 0.0
    hits=0
    for rec,rel in zip(recommendations,ground_truth):
        if set(rec[:k]) & set(rel):
            hits+=1
    return float(hits/len(recommendations))