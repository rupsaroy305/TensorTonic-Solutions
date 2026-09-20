import numpy as np

def stratified_split(X: list, y: list, test_size: float = 0.2, seed: int = 42) -> dict:
    X = np.array(X)
    y = np.array(y)
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = round(len(idx) * test_size)
        if len(idx) > 1:
            n_test = min(n_test, len(idx) - 1)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    return {
        "X_train": X[train_idx],
        "X_test": X[test_idx],
        "y_train": y[train_idx],
        "y_test": y[test_idx]
    }