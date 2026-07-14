def remove_stopwords(tokens, stopwords):
    s=set(stopwords)
    return [t for t in tokens if t not in s]