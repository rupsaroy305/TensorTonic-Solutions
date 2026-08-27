import numpy as np
def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    values=np.array(values,dtype=float)
    transitions=np.array(transitions,dtype=float)
    rewards=np.array(rewards,dtype=float)
    q=rewards+gamma*np.sum(transitions*values,axis=2)
    return np.max(q, axis=1).tolist()