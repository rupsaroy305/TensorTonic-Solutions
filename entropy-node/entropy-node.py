import numpy as np

def entropy_node(y):
    y=np.array(y)
    if len(y)==0:
        return 0.0
    _,c=np.unique(y,return_counts=True)
    p=c/len(y)
    e=0.0
    for i in p:
        if i>0:
            e-=i*np.log2(i)
    return float(e)