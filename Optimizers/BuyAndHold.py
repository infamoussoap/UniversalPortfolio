import numpy as np


def buy_and_hold(returns, w=None):
    returns = np.array(returns)
    T, N = returns.shape

    w = np.ones(N) / N if w is None else w
    
    weight_history = np.zeros(returns.shape)
    running_returns = np.zeros(T)

    for i in range(T):
        weight_history[i] = w

        running_returns[i] = w @ returns[i]

        w = w * returns[i, :]
        w = w / np.sum(w)

    return np.prod(running_returns), weight_history
