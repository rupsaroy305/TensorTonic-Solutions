import numpy as np

def silhouette_score(X, labels):
    import numpy as np
    X=np.asarray(X); labels=np.asarray(labels)
    d=np.linalg.norm(X[:,None]-X[None,:],axis=2)
    u=np.unique(labels)
    a=np.array([d[i,labels==labels[i]].sum()/(sum(labels==labels[i])-1) if sum(labels==labels[i])>1 else 0 for i in range(len(X))])
    b=np.array([min(d[i,labels==c].mean() for c in u if c!=labels[i]) for i in range(len(X))])
    return float(np.mean((b-a)/np.maximum(a,b)))