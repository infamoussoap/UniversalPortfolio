import numpy as np


def egd_tilde(returns, alpha, eta):
    returns = np.array(returns)

    T, N = returns.shape

    w = np.ones(N) / N
    running_returns = np.zeros(T)
    historic_weights = np.zeros((T, N))

    for i in range(T):
        historic_weights[i] = w

        x = returns[i, :]
        x_tilde = (1 - alpha / N) * x + (alpha / N)

        w_tilde = (1 - alpha) * w + (alpha / N)
        running_returns[i] = x @ w_tilde

        grad = x_tilde / (w @ x_tilde)

        w = w * np.exp(eta * grad)
        w = w / np.sum(w)

    return np.prod(running_returns), historic_weights
