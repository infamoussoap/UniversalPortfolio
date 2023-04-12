import numpy as np


def egd(returns, eta):
    returns = np.array(returns)

    T, N = returns.shape

    w = np.ones(N) / N
    running_returns = np.zeros(T)

    historic_weights = np.zeros((T, N))

    for i in range(T):
        x = returns[i, :]

        running_returns[i] = x @ w
        historic_weights[i] = w

        grad = x / (w @ x)

        w = w * np.exp(eta * grad)
        w = w / np.sum(w)

    return np.prod(running_returns), historic_weights
