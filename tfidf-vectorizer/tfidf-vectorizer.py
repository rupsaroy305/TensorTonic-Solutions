import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    docs=[d.lower().split() for d in documents]
    vocab=sorted({w for d in docs for w in d})
    if not vocab:return np.zeros((len(documents),0)),[]
    idx={w:i for i,w in enumerate(vocab)}
    N,V=len(docs),len(vocab)
    tf=np.zeros((N,V),float)
    df=np.zeros(V,int)
    for i,d in enumerate(docs):
        if not d:continue
        c={}
        for w in d:c[w]=c.get(w,0)+1
        for w,n in c.items():
            j=idx[w]
            tf[i,j]=n/len(d)
            df[j]+=1
    idf=np.log(N/df)
    return tf*idf,vocab