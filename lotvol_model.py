"""
Lotka-Volterra predator-prey model.

The model is:

```
x_m0 = (logH_m0, logL_m0) ~ pi(x_m0) \propto 1
logH_mt ~ N(logH_{m,t-1} + (alpha - beta exp(logL_{m,t-1})) dt/m,
            sigma_H^2 dt/m)
logL_mt ~ N(logL_{m,t-1} + (-gamma + delta exp(logH_{m,t-1})) dt/m,
            sigma_L^2 dt/m)
y_t ~ N(x_{m,mt}, diag(tau_H, tau_L))
```

- Model parameters: `theta = (alpha, beta, gamma, delta, sigma_H, sigma_L, tau_H, tau_L)`.
- Global constants: `dt` and `n_res`, i.e., `m`.
- State dimensions: `n_state = (n_res, 2)`.
- Measurement dimensions: `n_meas = 2`.  Note that `y_t` corresponds to `x_t = (x_{m,mt}, ..., x_{m,mt+(m-1)})`, i.e., aligns with the first element of `x_t`.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import lax


# --- helper functions ---------------------------------------------------------

def drift(x, dt, theta):
    """
    Calculates the SDE drift function.
    """
    alpha = theta[0]
    beta = theta[1]
    gamma = theta[2]
    delta = theta[3]
    return x + jnp.array([alpha - beta * jnp.exp(x[1]),
                          -gamma + delta * jnp.exp(x[0])]) * dt


def euler_sim(n_steps, x, dt, theta, key):
    """
    Euler simulation for `n_steps`.
    """
    sigma = theta[4:6] * jnp.sqrt(dt)

    # setup lax.scan:
    # scan function
    def fun(carry, t):
        key, subkey = random.split(carry["key"])
        x = drift(carry["x"], dt, theta) + \
            random.normal(subkey, (n_state[1],)) * sigma
        res = {"x": x, "key": key}
        return res, res
    # scan initial value
    init = {"x": x, "key": key}
    # lax.scan itself
    last, full = lax.scan(fun, init, jnp.arange(n_steps))
    return full["x"]


# --- main functions -----------------------------------------------------------

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
    dt_res = dt/n_res
    x0 = jnp.append(jnp.expand_dims(
        x_prev[n_res-1], axis=0), x_curr[:n_res-1], axis=0)
    x1 = x_curr
    sigma = theta[4:6] * jnp.sqrt(dt_res)
    lp = jax.vmap(lambda t:
                  jsp.stats.norm.logpdf(x1[t],
                                        loc=drift(x0[t], dt_res, theta),
                                        scale=sigma))(jnp.arange(n_res))
    return jnp.sum(lp)


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
    return euler_sim(n_res, x_prev[n_res-1], dt/n_res, theta, key)


def state_sample_for(x_prev, theta, key):
    """
    Samples from `x_curr ~ p(x_curr | x_prev, theta)`.

    For-loop version for testing.

    Args:
        x_prev: State variable at previous time `t-1`.
        theta: Parameter value.
        key: PRNG key.

    Returns:
        Sample of the state variable at current time `t`: `x_curr ~ p(x_curr | x_prev, theta)`.
    """
    dt_res = dt/n_res
    sigma = theta[4:6] * jnp.sqrt(dt_res)
    x_curr = jnp.zeros(n_state)
    x_state = x_prev[n_res-1]
    for t in range(n_res):
        key, subkey = random.split(key)
        x_state = drift(x_state, dt_res, theta) + \
            random.normal(subkey, (n_state[1],)) * sigma
        x_curr = x_curr.at[t].set(x_state)
    return x_curr


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
    tau = theta[6:8]
    return jnp.sum(
        jsp.stats.norm.logpdf(y_curr, loc=x_curr[0], scale=tau)
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
    tau = theta[6:8]
    return x_curr[0] + tau * random.normal(key, (n_state[1],))


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
    key, subkey = random.split(key)
    x_init = meas_sample(y_init, theta, subkey)
    x_step = euler_sim(n_res-1, x_init, dt/n_res, theta, key)
    return jnp.append(jnp.expand_dims(x_init, axis=0), x_step, axis=0)
