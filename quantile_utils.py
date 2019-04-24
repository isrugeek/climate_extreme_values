#!/usr/bin/env python
"""
Quantile Forest Experiment

The idea of this experiment is that, if we want to be able to predict the near
term maxima in some time series, it might be possible to learn the CDF over
future values associated with the currently observed window, and then take the
maximum of that CDF as the prediction for the max. We compare this approach
with the more direct "just predict the maximum" approach.
"""
import numpy as np


def simulate_ts(T=100, p=10, **kwargs):
    """
    Simulate TS with Covariates

    Simple linear model over X that evolve according to OU process.

    :param T: The length of the time series.
    :param p: The dimension of the predictors.
    :return (X, y): A tuple containing the covariates and time series value
      over time. X is T x p and y is length T.

    Example
    -------
    >>> X, y = simulate_ts()
    """
    X = np.zeros((T, p))
    for j in range(p):
        X[:, j] = sim_ou()

    beta = np.random.normal(size=(p,))
    y = np.dot(X, beta)
    return X, y


def sim_ou(T=100, mu=0, theta=0.1, sigma=0.1, delta_t=1):
    """
    Simulate an Orenstein Uhlenbeck Process

    Only reason we prefer this to a simple RW is that it has stationary mean
    and variance.

    Example
    -------
    >>> y = sim_ou(100, theta = .1)
    """
    y = np.zeros((T,))

    for i in range(1, T):
        prev = y[i - 1]
        eps_t = np.random.normal(scale=np.sqrt(delta_t))
        y[i] = prev + theta * (mu - prev) + sigma * eps_t

    return y


def extract_window(x, past, now, future):
    """
    Helper to split a time window
    """
    return x[past:now], x[now:future]


def windows(x, l_past=50, l_future=20, stride=5):
    """
    Strided Windows around a Timepoint

    This splits one long time series into many strided pieces. Each piece
    is futher split into two parts, a past and a future, around a central
    timepoint.

    :param x: A 1-dimensional time series, on which to extract small windows.
    :param l_past: The length of the sequences in the past component of each
      piece.
    :param l_future: The length of the sequence in the future component of each
      piece.
    :param stride: The number of timesteps to skip between windows.
    :return pasts, futures: A tuple of numpy arrays. Each row is a small time
      series, the i^th row of past corresponds to the i^th row of future (those
      are the next l_future) timepoints.

    Example
    -------
    >>> y = sim_ou(500, theta = .1)
    >>> windows(y)
    """
    T = x.size
    start_ix = np.arange(l_past, T - l_future, stride)
    pasts, futures = [], []

    for ix in start_ix:
        past, future = extract_window(x, ix - l_past, ix, ix + l_future)
        pasts.append(past)
        futures.append(future)

    return np.array(pasts), np.array(futures)


def window_samples(pasts, futures, q=[0.8, 0.95, 1]):
    """
    Extract Responses from Future Windows

    The future windows aren't directly usable for training purposes.

    :param pasts: A numpy array whose rows correspond to small subwindows from
      the past, as output by windows().
    :param futures: A numpy array whose values correspond to small subwindows
      from the future, as output by windows().
    :param q: A list giving the quantiles that we want to extract responses for
    :return samples: A list of dictionaries, each of which has a single response value
      y corresponding to some quantile ("quantile") and a vector of past values that
      can be used for prediction. The index of each unique x time window is maintained by
      the element "window_ix"

    Examples
    --------
    >>> y = sim_ou(500, theta = .1)
    >>> p, f = windows(y)
    >>> window_samples(p, f)
    """
    Q = np.quantile(futures, q=q, axis=1).T

    samples = []
    for i in range(pasts.shape[0]):
        for l in range(Q.shape[1]):
            samples.append({
                "x": pasts[i],
                "y": Q[i, l],
                "quantile": q[l],
                "window_ix": i
            })

    return samples
