import numpy as np

def bag_of_words_vector(tokens, vocab):
    d={w:i for i,w in enumerate(vocab)}
    a=np.zeros(len(vocab),int)
    for w in tokens:
        if w in d:a[d[w]]+=1
    return a