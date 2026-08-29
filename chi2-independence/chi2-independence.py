import numpy as np

def chi2_independence(C: list) -> dict:
    C=np.array(C,dtype=float)
    row=C.sum(axis=1)
    col=C.sum(axis=0)
    total=C.sum()
    expected=np.outer(row,col)/total
    chi2=np.sum((C-expected)** 2/expected)
    return {"chi2":float(chi2),"expected": expected}