import numpy as np

def roc_curve(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score, kind="stable")
    scores = y_score[order]
    labels = y_true[order]
    positives = np.sum(labels == 1)
    negatives = np.sum(labels == 0)
    group_end = np.r_[scores[:-1] != scores[1:], True]
    cum_tp = np.cumsum(labels == 1)
    cum_fp = np.cumsum(labels == 0)
    tp = cum_tp[group_end]
    fp = cum_fp[group_end]
    thresholds = scores[group_end]
    tpr = np.r_[0.0, tp / positives]
    fpr = np.r_[0.0, fp / negatives]
    thresholds = np.r_[np.inf, thresholds]
    return fpr, tpr, thresholds