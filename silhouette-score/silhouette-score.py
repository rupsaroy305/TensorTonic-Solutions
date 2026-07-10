import numpy as np

def silhouette_score(X, labels):
    X=np.asarray(X,float)
    labels=np.asarray(labels)
    d=np.sqrt(((X[:,None]-X[None,:])**2).sum(axis=2))
    s=[]
    for i in range(len(X)):
        same=labels==labels[i]
        same[i]=False
        if np.any(same):
            a=d[i,same].mean()
        else:
            a=0.0
        b=np.inf
        for c in np.unique(labels):
            if c==labels[i]:
                continue
            m=labels==c
            if np.any(m):
                b=min(b,d[i,m].mean())
        s.append((b-a)/max(a,b) if max(a,b)>0 else 0.0)
    return float(np.mean(s))