import numpy as np
def priority_replay_sample(priorities: list, alpha: float, beta: float) -> list:
    p=np.array(priorities, dtype=float)**alpha
    p=p/p.sum()
    w=(len(p)*p)**(-beta)
    w=w/w.max()
    return [p.tolist(), w.tolist()]