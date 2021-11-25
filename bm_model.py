"""
Brownian motion state space model.

The model is:

```
x_0 ~ pi(x_0) \propto 1
x_t ~ N(x_{t-1} + mu * dt, sigma * sqrt(dt))
y_t ~ N(x_t, tau)
```

The parameter values are `theta = (mu, sigma, tau)`, the meaurement and state dimensions are `n_meas = 1` and `n_state = 1`, and `dt` is a global constant.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random


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
    return jnp.squeeze(
        jsp.stats.norm.logpdf(x_curr, loc=x_prev + mu * dt,
                              scale=sigma * jnp.sqrt(dt))
    )


def state_sample(x_prev, theta, key):
    """
    Samples from `x_curr ~ p(x_curr | x_prev, theta)`.

    Args:
        x_prev: State variable at previous time `t-1`.
        theta: Parameter value.
        key: PRNG key.

    Returns:
        Sample of the state variable at current time `t`: `x_curr ~ p(x_curr | x_prev, theta)`.
    """
    mu = theta[0]
    sigma = theta[1]
    x_mean = x_prev + mu * dt
    x_sd = sigma * jnp.sqrt(dt)
    return x_mean + x_sd * random.normal(key=key)


def meas_lpdf(y_curr, x_curr, theta):
    """
    Log-density of `p(y_curr | x_curr, theta)`.

    Args:
        y_curr: Measurement variable at current time `t`.
        x_curr: State variable at current time `t`.
        theta: Parameter value.

    Returns
        The log-density of `p(y_curr | x_curr, theta)`.
    """
    tau = theta[2]
    return jnp.squeeze(
        jsp.stats.norm.logpdf(y_curr, loc=x_curr, scale=tau)
    )


def meas_sample(x_curr, theta, key):
    """
    Sample from `p(y_curr | x_curr, theta)`.

    Args:
        x_curr: State variable at current time `t`.
        theta: Parameter value.
        key: PRNG key.

    Returns:
        Sample of the measurement variable at current time `t`: `y_curr ~ p(y_curr | x_curr, theta)`.
    """
    tau = theta[2]
    return x_curr + tau * random.normal(key=key)


def init_logw(x_init, y_init, theta):
    """
    Log-weight of the importance sampler for initial state variable `x_init`.

    Suppose that 
    ```
    x_init ~ q(x_init) = q(x_init | y_init, theta)
    ```
    Then function returns
    ```
    logw = log p(y_init | x_init, theta) + log p(x_init | theta) - log q(x_init)
    ```

    Args:
        x_init: State variable at initial time `t = 0`.
        y_init: Measurement variable at initial time `t = 0`.
        theta: Parameter value.

    Returns:
        The log-weight of the importance sampler for `x_init`.
    """
    return jnp.zeros(())
    # return -meas_lpdf(x_init, y_init, theta)


def init_sample(y_init, theta, key):
    """
    Sampling distribution for initial state variable `x_init`. 

    Samples from an importance sampling proposal distribution
    ```
    x_init ~ q(x_init) = q(x_init | y_init, theta)
    ```
    See `init_logw()` for details.

    Args:
        y_init: Measurement variable at initial time `t = 0`.
        theta: Parameter value.
        key: PRNG key.

    Returns:
        Sample from the proposal distribution for `x_init`.
    """
    return meas_sample(y_init, theta, key)
