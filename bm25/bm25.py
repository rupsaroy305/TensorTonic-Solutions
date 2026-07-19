import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    if not docs:return np.array([],float)
    N=len(docs)
    dl=np.array([len(d) for d in docs],float)
    avgdl=dl.mean() if N else 0
    df={}
    for d in docs:
        for t in set(d):df[t]=df.get(t,0)+1
    tf=[Counter(d) for d in docs]
    s=np.zeros(N,float)
    for t in query_tokens:
        if t not in df:continue
        idf=np.log((N-df[t]+0.5)/(df[t]+0.5)+1)
        for i,c in enumerate(tf):
            f=c.get(t,0)
            if f:
                s[i]+=idf*f*(k1+1)/(f+k1*(1-b+b*dl[i]/avgdl))
    return s