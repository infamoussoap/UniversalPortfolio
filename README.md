# Convex optimization over a probability simplex
This repository contains the code for our paper [Convex optimization over a probability simplex](https://arxiv.org/abs/2305.09046). It has usable code for <ins>Universal Portfolios</ins> and reproducible code for this paper section.

Our other [repository](https://github.com/infamoussoap/ConvexHull) contains the code for <ins>Projection onto a Convex Hull</ins>, <ins>Optimal Question Weighting</ins>, 

## Universal Portfolios
A portfolio on the $t$-th day can be described as the portfolio vector $w^t\in\mathbb{R}^N$ with $\sum_i w^t_i = 1$ and $w_i\geq 0$, where $w_i^t$ represents the proportion of an investor's total wealth in asset $i$ at the beginning of the trading day.

Ideally, we would like to find a sequence of the portfolio vectors $w^t$ to maximize the total return over the portfolio's lifetime. However, this is too ambitious as the stock market returns are unknown beforehand. Instead, we consider a more restricted class of portfolios in which an investor must maintain a fixed weight in each asset over the portfolio lifetime. 

Let $u$ be the best fixed-weight portfolio vector. If one can generate a sequence of portfolio vectors $w^t$ such that it approaches the returns of $u$, then $(w^t)_t$ is known as a Universal Portfolio.

Our code introduces a new Universal Portfolio, which we call the Cauchy Simplex, as well as implements other Universal Portfolios like Exponentiated Gradient Descent by [Helmbold et al.](https://www.cis.upenn.edu/~mkearns/finread/helmbold98line.pdf) and Newton-based Method by [Agarwal et al.](https://dl.acm.org/doi/10.1145/1143844.1143846).

We also implement the Buy and Hold strategy, used as a benchmark.

## Datasets
Datasets are taken from http://www.cs.technion.ac.il/~rani/portfolios/, but are now unavailable. It is retrieved using the Wayback Machine: https://web.archive.org/web/20220111131743/http://www.cs.technion.ac.il/~rani/portfolios/
