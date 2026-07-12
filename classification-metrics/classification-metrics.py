import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    y_true=np.asarray(y_true)
    y_pred=np.asarray(y_pred)
    acc=float(np.mean(y_true==y_pred))
    labels=np.unique(np.concatenate((y_true,y_pred)))
    if average=="micro":
        tp=np.sum(y_true==y_pred)
        fp=np.sum(y_true!=y_pred)
        fn=fp
        p=tp/(tp+fp) if tp+fp else 0.0
        r=tp/(tp+fn) if tp+fn else 0.0
        f=2*p*r/(p+r) if p+r else 0.0
    else:
        ps,rs,fs,sup=[],[],[],[]
        for c in labels:
            tp=np.sum((y_true==c)&(y_pred==c))
            fp=np.sum((y_true!=c)&(y_pred==c))
            fn=np.sum((y_true==c)&(y_pred!=c))
            s=np.sum(y_true==c)
            p=tp/(tp+fp) if tp+fp else 0.0
            r=tp/(tp+fn) if tp+fn else 0.0
            f=2*p*r/(p+r) if p+r else 0.0
            ps.append(p); rs.append(r); fs.append(f); sup.append(s)

        if average=="macro":
            p=float(np.mean(ps)); r=float(np.mean(rs)); f=float(np.mean(fs))
        elif average=="weighted":
            w=np.array(sup)/np.sum(sup)
            p=float(np.sum(np.array(ps)*w))
            r=float(np.sum(np.array(rs)*w))
            f=float(np.sum(np.array(fs)*w))
        else:  # binary
            c=pos_label
            tp=np.sum((y_true==c)&(y_pred==c))
            fp=np.sum((y_true!=c)&(y_pred==c))
            fn=np.sum((y_true==c)&(y_pred!=c))
            p=tp/(tp+fp) if tp+fp else 0.0
            r=tp/(tp+fn) if tp+fn else 0.0
            f=2*p*r/(p+r) if p+r else 0.0
    return {"accuracy":acc,"precision":float(p),"recall":float(r),"f1":float(f)}