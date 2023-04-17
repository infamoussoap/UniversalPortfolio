import numpy as np


def cauchy_simplex_tilde(returns, beta, eta):
    returns = np.array(returns)

    T, N = returns.shape

    w = np.ones(N) / N

    running_returns = np.zeros(T)
    historic_weights = np.zeros((T, N))

    for i in range(T):
        w_tilde = (1 - beta) * w + (beta / N)

        historic_weights[i] = w_tilde

        x = returns[i, :]
        running_returns[i] = x @ w_tilde

        grad = - x / (w @ x)
        dw_dt = w * (grad - w @ grad)

        w = w - eta * dw_dt
        w = w / np.sum(w)

        assert np.all(w >= 0)

    return np.prod(running_returns), historic_weights
