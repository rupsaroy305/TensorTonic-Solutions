def f1_micro(y_true,y_pred):
    tp=fp=fn=0

    for t,p in zip(y_true,y_pred):
        if t==p:
            tp+=1
        else:
            fp+=1
            fn+=1

    d=2*tp+fp+fn
    if d==0:
        return 0.0

    return float((2*tp)/d)