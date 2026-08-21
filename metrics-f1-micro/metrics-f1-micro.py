def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    tp=sum(t==p for t,p in zip(y_true,y_pred))
    total=len(y_true)
    fp=total-tp
    fn=total-tp
    if 2*tp+fp+fn==0:
        return 0.0
    return round(2*tp/(2*tp+fp+fn),4)