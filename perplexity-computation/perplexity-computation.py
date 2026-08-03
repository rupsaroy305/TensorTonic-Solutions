import numpy as np

def perplexity(prob_distributions,actual_tokens):
    p=np.array([prob_distributions[i][actual_tokens[i]] for i in range(len(actual_tokens))],float)
    return float(np.exp(-np.mean(np.log(p))))