"""
Brownian motion state space model.

The model is:

```
x_0 ~ pi(x_0) \propto 1
x_t ~ N(x_{t-1} + mu * dt, sigma * sqrt(dt))
y_t ~ N(x_t, tau)
```

The parameter values are `theta = (mu, sigma, tau)`, and `dt` is a global constant.
"""

import numpy as np
import scipy as sp
import scipy.stats


# state-space dimensions
n_meas = 1
n_state = 1


def state_lpdf(x_curr, x_prev, theta):
    """
    Calculates the log-density of `p(x_curr | x_prev, theta)`.

    Args:
        x_curr: State variable at current time `t`.
        x_prev: State variable at previous time `t-1`.
        theta: Parameter value.

    Returns:
        The log-density of `p(x_curr | x_prev, theta)`.
    """
    mu = theta[0]
    sigma = theta[1]
    return sp.stats.norm.logpdf(x_curr, loc=x_prev + mu * dt, scale=sigma * np.sqrt(dt))


def state_sample(x_prev, theta):
    """
    Samples from `x_curr ~ p(x_curr | x_prev, theta)`.

    Args:
        x_prev: State variable at previous time `t-1`.
        theta: Parameter value.

    Returns:
        Sample of the state variable at current time `t`: `x_curr ~ p(x_curr | x_prev, theta)`.
    """
    mu = theta[0]
    sigma = theta[1]
    return sp.stats.norm.rvs(loc=x_prev + mu * dt, scale=sigma * np.sqrt(dt))


def meas_lpdf(y_curr, x_curr, theta):
    """
    Log-density of `p(y_curr | x_curr, theta)`.

    Args:
        y_curr: Measurement variable at current time `t`.
        x_curr: State variable at current time `t`.
        theta: Parameter value.

    Returns
        The log-density of `p(x_curr | x_prev, theta)`.
    """
    tau = theta[2]
    return sp.stats.norm.logpdf(y_curr, loc=x_curr, scale=tau)


def meas_sample(x_curr, theta):
    """
    Sample from `p(y_curr | x_curr, theta)`.

    Args:
        x_curr: State variable at current time `t`.
        theta: Parameter value.

    Returns:
        Sample of the measurement variable at current time `t`: `y_curr ~ p(y_curr | x_curr, theta)`.
    """
    tau = theta[2]
    return sp.stats.norm.rvs(loc=x_curr, scale=tau)
