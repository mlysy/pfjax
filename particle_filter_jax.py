"""
Particle filter in JAX.

Uses the same API as NumPy/SciPy version.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random


def meas_sim(n_obs, x_init, theta, key):
    """
    Simulate data from the state-space model.

    Args:
        n_obs: Number of observations to generate.
        x_init: Initial state value at time `t = 0`.
        theta: Parameter value.
        key: PRNG key.

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
                         a=jnp.arange(n_particles), shape=(n_particles,), p=prob)


def particle_filter(y_meas, theta, n_particles, key):
    """
    Apply particle filter for given value of `theta`.

    Closely follows Algorithm 2 of https://arxiv.org/pdf/1306.3277.pdf.

    FIXME: Uses a hard-coded prior for initial state variable `x_state[0]`.  Need to make this more general.

    Args:
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_1, ..., y_T)`, where `T = n_obs`.
        theta: Parameter value.
        n_particles: Number of particles.
        key: PRNG key.

    Returns:
        A dictionary with elements:
            - `X_particles`: An `ndarray` with leading dimensions `(n_obs, n_particles)` containing the state variable particles.
            - `logw_particles`: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.
            - `ancestor_particles`: An integer `ndarray` of shape `(n_obs, n_particles)` where each element gives the index of the particle's ancestor at the previous time point.  Since the first time point does not have ancestors, the first row of `ancestor_particles` contains all `-1`.
    """
    # memory allocation
    n_obs = y_meas.shape[0]
    X_particles = jnp.zeros((n_obs, n_particles, n_state))
    logw_particles = jnp.zeros((n_obs, n_particles))
    ancestor_particles = jnp.zeros((n_obs, n_particles), dtype=int)
    # initial particles have no ancestors
    ancestor_particles = ancestor_particles.at[0].set(-1)
    # initial time point
    # FIXME: Hard-coded flat prior on x_0.  Make this more general.
    key, *subkeys = random.split(key, num=n_particles+1)
    X_particles = X_particles.at[0].set(
        jax.vmap(lambda k: meas_sample(y_meas[0], theta, k))(
            jnp.array(subkeys)
        )
    )
    # sample directly from posterior p(x_0 | y_0, theta)
    logw_particles = logw_particles.at[0].set(0.)
    # subsequent time points
    for t in range(1, n_obs):
        # resampling step
        key, subkey = random.split(key)
        ancestor_particles = ancestor_particles.at[t].set(
            particle_resample(logw_particles[t-1], subkey)
        )
        # update
        key, *subkeys = random.split(key, num=n_particles+1)
        X_particles = X_particles.at[t].set(
            jax.vmap(lambda xs, k: state_sample(xs, theta, k))(
                X_particles[t-1, ancestor_particles[t]], jnp.array(subkeys)
            )
        )
        logw_particles = logw_particles.at[t].set(
            jnp.squeeze(
                jax.vmap(lambda xs: meas_lpdf(y_meas[t], xs, theta))(
                    X_particles[t]
                )
            )
        )
    return {
        "X_particles": X_particles,
        "logw_particles": logw_particles,
        "ancestor_particles": ancestor_particles
    }
