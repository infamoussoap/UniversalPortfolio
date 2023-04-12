import numpy as np


def cauchy_simplex(returns, gamma):
    returns = np.array(returns)

    T, N = returns.shape

    w = np.ones(N) / N

    running_returns = np.zeros(T)
    historic_weights = np.zeros((T, N))

    for i in range(T):
        historic_weights[i] = w

        x = returns[i, :]
        running_returns[i] = x @ w

        grad = - x / (w @ x)
        dw_dt = w * (grad - w @ grad)

        w = w - gamma * dw_dt
        w = w / np.sum(w)

        assert np.all(w >= 0)

    return np.prod(running_returns), historic_weights

