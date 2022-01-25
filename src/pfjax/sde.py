"""
Generic methods for SDEs.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import lax


def euler_sim_diag(n_steps, x, dt, drift, diff, theta, key):
    """
    Simulate SDE with diagonal diffusion using Euler-Maruyama discretization.

    Args:
        n_steps: Number of steps to simulate.
        x: Initial value of the SDE.  A vector of size `n_dims`.
        dt: Interobservation time.
        drift: Drift function having signature `drift(x, theta)` and returning a vector of size `n_dims`.
        diff: Diffusion function having signature `diff(x, theta)` and returning a vector of size `n_dims`.
        theta: Parameter value.
        key: PRNG key.

    Returns:
        Simulated SDE values in a matrix of size `n_steps x n_dims`.
    """

    # setup lax.scan:
    # scan function
    def fun(carry, t):
        key, subkey = random.split(carry["key"])
        x = carry["x"]
        dr = x + drift(x, dt, theta) * dt
        df = diff(x, theta) * jnp.sqrt(dt)
        x = dr + df * random.normal(subkey, (x.shape[0],))
        res = {"x": x, "key": key}
        return res, res
    # scan initial value
    init = {"x": x, "key": key}
    # lax.scan itself
    last, full = lax.scan(fun, init, jnp.arange(n_steps))
    return full["x"]


def euler_lpdf_diag(x, dt, drift, diff, theta):
    """
    Calculate the log PDF of observations from an SDE with diagonal diffusion using the Euler-Maruyama discretization.

    Args:
        x: SDE observations.  An array of size `n_obs x n_dims`.
        dt: Interobservation time.
        drift: Drift function having signature `drift(x, theta)` and returning a vector of size `n_dims`.
        diff: Diffusion function having signature `diff(x, theta)` and returning a vector of size `n_dims`.
        theta: Parameter value.

    Returns:
        The log-density of the SDE observations.
    """
    x0 = x[:-1, :]
    x1 = x[1:, :]
    lp = jax.vmap(lambda t:
                  jsp.stats.norm.logpdf(
                      x=x1[t],
                      loc=x0[t] + drift(x0[t], theta) * dt,
                      scale=diff(x0[t], theta) * jnp.sqrt(dt)
                  ))(jnp.arange(x0.shape[0]))
    return jnp.sum(lp)
