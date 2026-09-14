def discount_returns(rewards: list, gamma: float) -> list:
    returns = [0.0] * len(rewards)
    returns[-1] = float(rewards[-1])
    for t in range(len(rewards) - 2, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t + 1]
    return returns