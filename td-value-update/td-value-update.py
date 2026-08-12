import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    V_new = np.array(V, dtype=float, copy=True)
    target = r + gamma * V_new[s_next]
    delta = target - V_new[s]
    V_new[s] += alpha * delta
    return V_new