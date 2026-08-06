def precision_recall_at_k(recommended, relevant, k):
    h=sum(i in set(relevant) for i in recommended[:k])
    return [h/k,h/len(relevant)]