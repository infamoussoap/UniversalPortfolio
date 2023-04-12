import numpy as np
import sys


def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def relative_entropy(u, v):
    mask = u > 0
    return u[mask] @ np.log(u[mask] / v[mask])


def best_returns(returns, w=None, verbose=False, gamma=0.01, max_iter=500):
    returns = np.array(returns)

    T, N = returns.shape

    if w is None:
        w = np.ones(N) / N

    current_returns = np.prod(returns @ w)
    for i in range(max_iter):
        w_old = w.copy()

        grad = np.sum(returns / (returns @ w)[:, None], axis=0)
        dw_dt = w * (grad - w @ grad)

        diff = np.max(grad) - w @ grad
        max_learning_rate = 1 / diff if diff > 1e-5 else 1e5
        learning_rate = gamma * max_learning_rate

        w = w + learning_rate * dw_dt
        w[w < 1e-10] = 0
        w = w / np.sum(w)

        new_returns = np.prod(returns @ w)
        if new_returns < current_returns:
            return w_old
        else:
            current_returns = new_returns

        assert np.all(w >= 0)

        if verbose:
            sys.stdout.write(f"{i}: {new_returns} \r")
            sys.stdout.flush()
    return w
