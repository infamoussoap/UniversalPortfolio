import numpy as np


def newton(returns, eta, beta, delta):
    returns = np.array(returns)

    T, N = returns.shape

    running_returns = np.zeros(T)
    historic_weights = np.zeros((T, N))

    I = np.eye(N)
    p = np.ones(N) / N

    hessian_total = 0
    grad_total = 0
    for i in range(T):
        r = returns[i, :]
        p_tilde = (1 - eta) * p + eta / N

        running_returns[i] = p_tilde @ r
        historic_weights[i] = p_tilde

        grad_total += log_grad(p, r)
        hessian_total += log_hessian(p, r)

        b = (1 + 1 / beta) * grad_total
        A = I - hessian_total

        q = delta * np.linalg.inv(A) @ b
        p = project(q, A)

    return np.prod(running_returns), historic_weights


def log_hessian(w, r):
    return - r[:, None] * r[None, :] / ((w @ r) ** 2)


def log_grad(w, r):
    return r / (w @ r)


def loss(q, p, A):
    return (q - p) @ A @ (q - p)


def project(q, A, max_iter=100, gamma=0.05):
    N = len(q)

    w = np.ones(N) / N
    current_loss = loss(q, w, A)

    for _ in range(max_iter):
        w_old = w.copy()

        grad = - A @ (q - w) - (q - w) @ A
        max_learning_rate = 1 / (np.max(grad) - w @ grad)

        eta = gamma * max_learning_rate
        Pi = grad - w @ grad
        w = w * (1 - eta * Pi)

        w = w / np.sum(w)

        new_loss = loss(q, w, A)

        if new_loss > current_loss:
            return w_old
        else:
            current_loss = new_loss

    return w

