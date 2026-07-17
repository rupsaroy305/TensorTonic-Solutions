import numpy as np
def perplexity(prob_distributions, actual_tokens):
    p=np.asarray(prob_distributions,dtype=float)
    t=np.asarray(actual_tokens)
    return float(np.exp(-np.mean(np.log(p[np.arange(len(t)),t]))))