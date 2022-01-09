"""
Particle filter in JAX.

This is the all for-loops + globals version for testing.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import lax
from jax.experimental.maps import xmap
from functools import partial


def simulate_for(n_obs, x_init, theta, key):
    """
    Simulate data from the state-space model.

    This is the testing version which uses a for-loop.

    Args:
        n_obs: Number of observations to generate.
        x_init: Initial state value at time `t = 0`.
        theta: Parameter value.
        key: PRNG key.

    Returns:
        y_meas: The sequence of measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        x_state: The sequence of state variables `x_state = (x_0, ..., x_T)`, where `T = n_obs-1`.
    """
    y_meas = jnp.zeros((n_obs, n_meas))
    x_state = jnp.zeros((n_obs, n_state))
    x_state = x_state.at[0].set(x_init)
    # initial observation
    key, subkey = random.split(key)
    y_meas = y_meas.at[0].set(meas_sample(x_init, theta, subkey))
    for t in range(1, n_obs):
        key, *subkeys = random.split(key, num=3)
        x_state = x_state.at[t].set(
            state_sample(x_state[t-1], theta, subkeys[0])
        )
        y_meas = y_meas.at[t].set(
            meas_sample(x_state[t], theta, subkeys[1])
        )
    return y_meas, x_state


# @partial(jax.jit, static_argnums=0)
def simulate(n_obs, x_init, theta, key):
    """
    Simulate data from the state-space model.

    Args:
        n_obs: Number of observations to generate.
        x_init: Initial state value at time `t = 0`.
        theta: Parameter value.
        key: PRNG key.

    Returns:
        y_meas: The sequence of measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        x_state: The sequence of state variables `x_state = (x_0, ..., x_T)`, where `T = n_obs-1`.
    """
    # lax.scan setup
    # scan function
    def fun(carry, x):
        key, *subkeys = random.split(carry["key"], num=3)
        x_state = state_sample(carry["x_state"], theta, subkeys[0])
        y_meas = meas_sample(x_state, theta, subkeys[1])
        res = {"y_meas": y_meas, "x_state": x_state, "key": key}
        return res, res
    # scan initial value
    key, subkey = random.split(key)
    init = {
        "y_meas": meas_sample(x_init, theta, subkey),
        "x_state": x_init,
        "key": key
    }
    # scan itself
    last, full = lax.scan(fun, init, jnp.arange(1, n_obs))
    # append initial values
    x_state = jnp.append(jnp.expand_dims(init["x_state"], axis=0),
                         full["x_state"], axis=0)
    y_meas = jnp.append(jnp.expand_dims(init["y_meas"], axis=0),
                        full["y_meas"], axis=0)
    return y_meas, x_state


def particle_resample(logw, key):
    """
    Particle resampler.

    This basic one just does a multinomial sampler, i.e., sample with replacement proportional to weights.

    Args:
        logw: Vector of `n_particles` unnormalized log-weights.
        key: PRNG key.

    Returns:
        Vector of `n_particles` integers between 0 and `n_particles-1`, sampled with replacement with probability vector `exp(logw) / sum(exp(logw))`.
    """
    wgt = jnp.exp(logw - jnp.max(logw))
    prob = wgt / jnp.sum(wgt)
    n_particles = logw.size
    return random.choice(key,
                         a=jnp.arange(n_particles),
                         shape=(n_particles,), p=prob)


def particle_filter_for(y_meas, theta, n_particles, key):
    """
    Apply particle filter for given value of `theta`.

    Closely follows Algorithm 2 of https://arxiv.org/pdf/1306.3277.pdf.

    This is the original implementation with for-loops.

    Args:
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        n_particles: Number of particles.
        key: PRNG key.

    Returns:
        A dictionary with elements:
            - `x_particles`: An `ndarray` with leading dimensions `(n_obs, n_particles)` containing the state variable particles.
            - `logw`: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.
            - `ancestors`: An integer `ndarray` of shape `(n_obs, n_particles)` where each element gives the index of the particle's ancestor at the previous time point.  Since the first time point does not have ancestors, the first row of `ancestors` contains all `-1`.
    """
    # memory allocation
    n_obs = y_meas.shape[0]
    x_particles = jnp.zeros((n_obs, n_particles, n_state))
    logw = jnp.zeros((n_obs, n_particles))
    ancestors = jnp.zeros((n_obs, n_particles), dtype=int)
    # initial particles have no ancestors
    ancestors = ancestors.at[0].set(-1)
    # initial time point
    key, *subkeys = random.split(key, num=n_particles+1)
    for p in range(n_particles):
        x_particles = x_particles.at[0, p].set(
            init_sample(y_meas[0], theta, subkeys[p])
        )
        logw = logw.at[0, p].set(
            init_logw(x_particles[0, p], y_meas[0], theta)
        )
    # subsequent time points
    for t in range(1, n_obs):
        # resampling step
        key, subkey = random.split(key)
        ancestors = ancestors.at[t].set(
            particle_resample(logw[t-1], subkey)
        )
        # update
        key, *subkeys = random.split(key, num=n_particles+1)
        for p in range(n_particles):
            x_particles = x_particles.at[t, p].set(
                state_sample(x_particles[t-1, ancestors[t, p]],
                             theta, subkeys[p])
            )
            logw = logw.at[t, p].set(
                meas_lpdf(y_meas[t], x_particles[t, p], theta)
            )
    return {
        "x_particles": x_particles,
        "logw": logw,
        "ancestors": ancestors
    }


@partial(jax.jit, static_argnums=2)
def particle_filter(y_meas, theta, n_particles, key):
    """
    Apply particle filter for given value of `theta`.

    Closely follows Algorithm 2 of https://arxiv.org/pdf/1306.3277.pdf.

    Args:
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        n_particles: Number of particles.
        key: PRNG key.

    Returns:
        A dictionary with elements:
            - `x_particles`: An `ndarray` with leading dimensions `(n_obs, n_particles)` containing the state variable particles.
            - `logw`: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.
            - `ancestors`: An integer `ndarray` of shape `(n_obs, n_particles)` where each element gives the index of the particle's ancestor at the previous time point.  Since the first time point does not have ancestors, the first row of `ancestors` contains all `-1`.
    """
    n_obs = y_meas.shape[0]

    # lax.scan setup
    # scan function
    def fun(carry, t):
        # resampling step
        key, subkey = random.split(carry["key"])
        ancestors = particle_resample(carry["logw"],
                                      subkey)
        # update particles
        key, *subkeys = random.split(key, num=n_particles+1)
        x_particles = jax.vmap(lambda xs, k: state_sample(xs, theta, k))(
            carry["x_particles"][ancestors], jnp.array(subkeys)
        )
        # update log-weights
        logw = jax.vmap(lambda xs: meas_lpdf(y_meas[t], xs, theta))(
            x_particles
        )
        # output
        res = {
            "ancestors": ancestors,
            "logw": logw,
            "x_particles": x_particles,
            "key": key
        }
        return res, res
    # scan initial value
    key, *subkeys = random.split(key, num=n_particles+1)
    # vmap version
    x_particles = jax.vmap(
        lambda k: init_sample(y_meas[0], theta, k))(jnp.array(subkeys))
    logw = jax.vmap(
        lambda xs: init_logw(xs, y_meas[0], theta))(x_particles)
    # xmap version: experimental!
    # x_particles = xmap(
    #     lambda ym, th, k: init_sample(ym, th, k),
    #     in_axes=([...], [...], ["particles", ...]),
    #     out_axes=["particles", ...])(y_meas[0], theta, jnp.array(subkeys))
    # logw = xmap(
    #     lambda xs, ym, th: init_logw(xs, ym, th),
    #     in_axes=(["particles", ...], [...], [...]),
    #     out_axes=["particles", ...])(x_particles, y_meas[0], theta)
    init = {
        "x_particles": x_particles,
        "logw": logw,
        "ancestors": -jnp.ones(n_particles, dtype=int),
        "key": key
    }
    # lax.scan itself
    last, full = lax.scan(fun, init, jnp.arange(1, n_obs))
    # append initial values
    out = {
        k: jnp.append(jnp.expand_dims(init[k], axis=0), full[k], axis=0)
        for k in ["x_particles", "logw", "ancestors"]
    }
    return out


def particle_loglik(logw):
    """
    Calculate particle filter marginal loglikelihood.

    FIXME: Libbi paper does `logmeanexp` instead of `logsumexp`...

    Args:
        logw: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.

    Returns:
        Particle filter approximation of
        ```
        log p(y_meas | theta) = log int p(y_meas | x_state, theta) * p(x_state | theta) dx_state
        ```
    """
    return jnp.sum(jsp.special.logsumexp(logw, axis=1))
