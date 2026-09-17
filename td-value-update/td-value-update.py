import numpy as np

def td_value_update(V: list, s: int, r: float, s_next: int, alpha: float, gamma: float) -> np.ndarray:
    V_new = np.array(V, dtype=float, copy=True)
    td_error = r + gamma * V_new[s_next] - V_new[s]
    V_new[s] += alpha * td_error
    return V_new