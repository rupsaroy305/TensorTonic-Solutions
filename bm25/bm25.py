import math
from collections import Counter
import numpy as np

def bm25_score(query_tokens: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    N = len(docs)
    avgdl = np.mean([len(doc) for doc in docs])

    query_terms = set(query_tokens)
    df = Counter()

    for doc in docs:
        for term in set(doc):
            df[term] += 1

    scores = []

    for doc in docs:
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0

        for term in query_terms:
            if term not in tf:
                continue

            idf = np.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)

            score += (
                idf * tf[term] * (k1 + 1)
                / (
                    tf[term]
                    + k1 * (1 - b + b * dl / avgdl)
                )
            )

        scores.append(score)

    return np.array(scores, dtype=float)