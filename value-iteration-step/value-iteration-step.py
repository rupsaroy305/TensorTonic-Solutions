import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    values = np.asarray(values, dtype=float)
    transitions = np.asarray(transitions, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    future = np.sum(transitions * values, axis=2)
    q_values = rewards + gamma * future
    new_values = np.max(q_values, axis=1)
    return new_values.tolist()