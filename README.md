# Convex optimization over a probability simplex
This repository contains the code for our paper [Convex optimization over a probability simplex](https://arxiv.org/abs/2305.09046). It has usable code for <ins>Universal Portfolios</ins> and reproducible code for this paper section.

Our other [repository](https://github.com/infamoussoap/ConvexHull) contains the code for <ins>Projection onto a Convex Hull</ins>, <ins>Optimal Question Weighting</ins>, 

## Universal Portfolios
Consider an investor with a fixed-time trading horizon, $T$, managing a portfolio of $N$ assets. Define the price relative for the $i$-th stock at time $t$ as $x^t_i = C^t_i / C^{t-1}_i$, where $C^t_i$ is the closing price at time $t$ for the $i$-th stock. So today's closing price of asset $i$ equals $x_i^t$ times yesterday's closing price, \textit{i.e.} today's price relative to yesterday's.

A portfolio at dat $t$ can be described as $w^t\in\mathbb{R}^N$ with $\sum_i w^t_i = 1$ and $w^t_i \geq 0$, where $w_i^t$ is the proportion of an investor's total wealth in asset $i$ at the beginning of the trading day. Then the wealth of the portfolio at the beginning of day $t+1$ is $w^t\cdot x^t$ times the wealth of the portfolio at day $t$.

Consider the average log-return of the portfolio
$$\frac{1}{T}\log\left(\prod_{t=1}^T w^t \cdot x^t\right) = \frac{1}{T}\sum_{t=1}^T\log(w^t\cdot x^t).$$
It is too ambitious to find a sequence of portfolio vectors $w^t$ that maximizes the average log-return. Instead, we wish to find such a sequence that approaches the best fixed-weight portfolio, <i>i.e.</i>
$$\frac{1}{T} LR_T\to 0, \quad \text{where}\quad LR_T = \sum_{t=1}^T\log(u\cdot x^t) - \sum_{t=1}^T\log(w^t\cdot x^t)$$
as $T\to\infty$, for some $u\in\mathbb{R}^N$ with $\sum_i u_i = 1$ and $u_i \geq 0$. 

If such a sequence can be found, $\{w^t\}_t$ is a universal portfolio. $LR_T$ is commonly known as the log-regret.

## Implemented Algorithms
For Universal Portfolios, we implement Cauchy-Simplex, Exponentiated Gradient Descent, Buy and Hold, and Newton-based Method by [Agarwal et al.](https://dl.acm.org/doi/10.1145/1143844.1143846).


## Datasets
Datasets are taken from http://www.cs.technion.ac.il/~rani/portfolios/, but are now unavailable. It is retrieved using the Wayback Machine: https://web.archive.org/web/20220111131743/http://www.cs.technion.ac.il/~rani/portfolios/
