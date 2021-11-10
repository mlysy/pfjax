"""
Particle filter in JAX.

Uses the same API as NumPy/SciPy version.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random

# --- model-specific functions and constants -----------------------------------

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
    return jsp.stats.norm.logpdf(x_curr, loc=x_prev + mu * dt, scale=sigma * jnp.sqrt(dt))


def state_sample(x_prev, theta, key):
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
        The log-density of `p(x_curr | x_prev, theta)`.
    """
    tau = theta[2]
    return jsp.stats.norm.logpdf(y_curr, loc=x_curr, scale=tau)


def meas_sample(x_curr, theta, key):
    """
    Sample from `p(y_curr | x_curr, theta)`.

    Args:
        x_curr: State variable at current time `t`.
        theta: Parameter value.

    Returns:
        Sample of the measurement variable at current time `t`: `y_curr ~ p(y_curr | x_curr, theta)`.
    """
    tau = theta[2]
    return x_curr + tau * random.normal(key=key)


# --- pf functions -------------------------------------------------------------

def meas_sim(n_obs, x_init, theta, key):
    """
    Simulate data from the state-space model.

    Args:
        n_obs: Number of observations to generate.
        x_init: Initial state value at time `t = 0`.
        theta: Parameter value.

    Returns:
        y_meas: The sequence of measurement variables `y_meas = (y_1, ..., y_T)`, where `T = n_obs`.
        x_state: The sequence of state variables `x_state = (x_1, ..., x_T)`, where `T = n_obs`.
    """
    y_meas = jnp.zeros((n_obs, n_meas))
    x_state = jnp.zeros((n_obs, n_state))
    x_prev = x_init
    for t in range(n_obs):
        key, *subkeys = random.split(key, num=3)
        x_state = x_state.at[t].set(state_sample(x_prev, theta, subkeys[0]))
        y_meas = y_meas.at[t].set(meas_sample(x_state[t], theta, subkeys[1]))
        x_prev = x_state[t]
    return y_meas, x_state


# --- tests --------------------------------------------------------------------

key = random.PRNGKey(0)

# parameter values
mu = 5
sigma = 1
tau = .1
theta = jnp.array([mu, sigma, tau])

print(theta)

# data specification
dt = .1
n_obs = 5
x_init = jnp.array([0.])

# simulate data

key, subkey = random.split(key)
y_meas, x_state = meas_sim(n_obs, x_init, theta, subkey)

print("y_meas = \n", y_meas)
print("x_state = \n", x_state)
