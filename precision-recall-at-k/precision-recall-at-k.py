def precision_recall_at_k(recommended,relevant,k):
    top=recommended[:k]
    rel=set(relevant)
    hits=0
    for x in top:
        if x in rel:
            hits+=1
    precision=hits/k if k>0 else 0.0
    recall=hits/len(rel) if len(rel)>0 else 0.0
    return [float(precision),float(recall)]