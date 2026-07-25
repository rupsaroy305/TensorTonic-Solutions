import numpy as np

def gini_impurity(y_left, y_right):
    y_left=np.asarray(y_left)
    y_right=np.asarray(y_right)
    nl,nr=len(y_left),len(y_right)
    n=nl+nr
    if n==0:return 0.0
    def g(y):
        if len(y)==0:return 0.0
        _,c=np.unique(y,return_counts=True)
        p=c/len(y)
        return 1-np.sum(p*p)
    return float((nl/n)*g(y_left)+(nr/n)*g(y_right))