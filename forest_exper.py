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
import matplotlib.pyplot as plt
from skgarden import RandomForestQuantileRegressor
from sklearn.ensemble import RandomForestRegressor

_, y = simulate_ts(T=1500)

pasts, futures = windows(y)
samples = window_samples(pasts, futures, np.arange(0, 1.05, 0.05))

###############################################################################
# First the quantile forest idea
###############################################################################
model = RandomForestQuantileRegressor(n_estimators=1000)

# fit model using all the quantiles
y_p = np.array([v["x"] for v in samples])
y_f = np.array([v["y"] for v in samples])
model.fit(y_p, y_f)

# make predictions only for 0.9 quantiles
x_q = np.array([v["x"] for v in samples if v["quantile"] == 1])
q_hat = model.predict(x_q, quantile=1)

y_q = np.array([v["y"] for v in samples if v["quantile"] == 1])
plt.scatter(y_q, q_hat)
#plt.show()

###############################################################################
# Next the random forest on just the maxima
###############################################################################
model = RandomForestRegressor(n_estimators=1000)
model.fit(x_q, y_q)

q_hat = model.predict(x_q)
plt.scatter(y_q, q_hat)

#plt.show() # direct prediction is wayyyy better

