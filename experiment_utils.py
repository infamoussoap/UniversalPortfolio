import numpy as np

from Optimizers import cauchy_simplex, egd, buy_and_hold


def evaluate_returns(price_relatives, risk_free=0.04):
    return (annualized_percentage_yield(price_relatives),
            sharpe_ratio(price_relatives, risk_free=risk_free))


def annualized_percentage_yield(price_relatives):
    price_relatives = np.array(price_relatives)

    y = len(price_relatives) // 252

    cumulative_wealth = np.prod(price_relatives)
    APY = pow(cumulative_wealth, 1 / y) - 1

    return APY


def sharpe_ratio(price_relatives, risk_free=0.04):
    price_relatives = np.array(price_relatives)
    APY = annualized_percentage_yield(price_relatives)

    returns = price_relatives - 1
    return (APY - risk_free) / np.std(returns)


def run_cs(returns):
    """ returns assumed to be the price relatives """
    returns = np.array(returns)

    T, N = returns.shape

    rescaled_returns = returns / np.max(returns, axis=1)[:, None]
    a = np.min(rescaled_returns)

    cs_eta = a * np.sqrt(np.log(N)) / (a * np.sqrt(np.log(N)) + np.sqrt(T))
    _, cs_history = cauchy_simplex(rescaled_returns, cs_eta)

    return cs_history


def run_egd(returns):
    """ returns assumed to be the price relatives """
    returns = np.array(returns)

    T, N = returns.shape

    rescaled_returns = returns / np.max(returns, axis=1)[:, None]
    a = np.min(rescaled_returns)

    egd_eta = 2 * a * np.sqrt(2 * np.log(N) / T)
    _, egd_history = egd(rescaled_returns, egd_eta)

    return egd_history


def run_hold(returns):
    """ returns assumed to be the price relatives """
    returns = np.array(returns)

    T, N = returns.shape

    rescaled_returns = returns / np.max(returns, axis=1)[:, None]

    _, hold_history = buy_and_hold(rescaled_returns)

    return hold_history
