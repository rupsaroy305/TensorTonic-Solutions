def target_encoding(categories, targets):
    sums={}
    counts={}
    for c,t in zip(categories,targets):
        sums[c]=sums.get(c,0)+t
        counts[c]=counts.get(c,0)+1
    return [float(sums[c]/counts[c]) for c in categories]