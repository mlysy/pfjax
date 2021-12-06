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
    key, *subkeys = random.split(key, num=n_particles+1)
    for p in range(n_particles):
        X_particles = X_particles.at[0, p].set(
            init_sample(y_meas[0], theta, subkeys[p])
        )
        logw_particles = logw_particles.at[0, p].set(
            init_logw(X_particles[0, p], y_meas[0], theta)
        )
    # X_particles = X_particles.at[0].set(
    #     jax.vmap(lambda k: init_sample(y_meas[0], theta, k))(
    #         jnp.array(subkeys)
    #     )
    # )
    # logw_particles = logw_particles.at[0].set(
    #     jax.vmap(lambda xs: init_logw(xs, y_meas[0], theta) +
    #              meas_lpdf(y_meas[0], xs, theta))(X_particles[0])
    # )
    # subsequent time points
    for t in range(1, n_obs):
        # resampling step
        key, subkey = random.split(key)
        ancestor_particles = ancestor_particles.at[t].set(
            particle_resample(logw_particles[t-1], subkey)
        )
        # update
        key, *subkeys = random.split(key, num=n_particles+1)
        for p in range(n_particles):
            X_particles = X_particles.at[t, p].set(
                state_sample(X_particles[t-1, ancestor_particles[t, p]],
                             theta, subkeys[p])
            )
            logw_particles = logw_particles.at[t, p].set(
                meas_lpdf(y_meas[t], X_particles[t, p], theta)
            )
        # X_particles = X_particles.at[t].set(
        #     jax.vmap(lambda xs, k: state_sample(xs, theta, k))(
        #         X_particles[t-1, ancestor_particles[t]], jnp.array(subkeys)
        #     )
        # )
        # logw_particles = logw_particles.at[t].set(
        #     jax.vmap(lambda xs: meas_lpdf(y_meas[t], xs, theta))(
        #         X_particles[t]
        #     )
        # )
    return {
        "X_particles": X_particles,
        "logw_particles": logw_particles,
        "ancestor_particles": ancestor_particles
    }


# @partial(jax.jit, static_argnums=2)
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
            - `X_particles`: An `ndarray` with leading dimensions `(n_obs, n_particles)` containing the state variable particles.
            - `logw_particles`: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.
            - `ancestor_particles`: An integer `ndarray` of shape `(n_obs, n_particles)` where each element gives the index of the particle's ancestor at the previous time point.  Since the first time point does not have ancestors, the first row of `ancestor_particles` contains all `-1`.
    """
    n_obs = y_meas.shape[0]

    # lax.scan setup
    # scan function
    def fun(carry, t):
        # resampling step
        key, subkey = random.split(carry["key"])
        ancestor_particles = particle_resample(carry["logw_particles"],
                                               subkey)
        # update particles
        key, *subkeys = random.split(key, num=n_particles+1)
        X_particles = jax.vmap(lambda xs, k: state_sample(xs, theta, k))(
            carry["X_particles"][ancestor_particles], jnp.array(subkeys)
        )
        # update log-weights
        logw_particles = jax.vmap(lambda xs: meas_lpdf(y_meas[t], xs, theta))(
            X_particles
        )
        # output
        res = {
            "ancestor_particles": ancestor_particles,
            "logw_particles": logw_particles,
            "X_particles": X_particles,
            "key": key
        }
        return res, res
    # scan initial value
    key, *subkeys = random.split(key, num=n_particles+1)
    # vmap version
    X_particles = jax.vmap(
        lambda k: init_sample(y_meas[0], theta, k))(jnp.array(subkeys))
    logw_particles = jax.vmap(
        lambda xs: init_logw(xs, y_meas[0], theta))(X_particles)
    # xmap version: experimental!
    # X_particles = xmap(
    #     lambda ym, th, k: init_sample(ym, th, k),
    #     in_axes=([...], [...], ["particles", ...]),
    #     out_axes=["particles", ...])(y_meas[0], theta, jnp.array(subkeys))
    # logw_particles = xmap(
    #     lambda xs, ym, th: init_logw(xs, ym, th),
    #     in_axes=(["particles", ...], [...], [...]),
    #     out_axes=["particles", ...])(X_particles, y_meas[0], theta)
    init = {
        "X_particles": X_particles,
        "logw_particles": logw_particles,
        "ancestor_particles": -jnp.ones(n_particles, dtype=int),
        "key": key
    }
    # lax.scan itself
    last, full = lax.scan(fun, init, jnp.arange(1, n_obs))
    # append initial values
    out = {
        k: jnp.append(jnp.expand_dims(init[k], axis=0), full[k], axis=0)
        for k in ["X_particles", "logw_particles", "ancestor_particles"]
    }
    return out


def particle_loglik(logw_particles):
    """
    Calculate particle filter marginal loglikelihood.

    FIXME: Libbi paper does `logmeanexp` instead of `logsumexp`...

    Args:
        logw_particles: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.

    Returns:
        Particle filter approximation of
        ```
        log p(y_meas | theta) = log int p(y_meas | x_state, theta) * p(x_state | theta) dx_state
        ```
    """
    return jnp.sum(jsp.special.logsumexp(logw_particles, axis=1))


def get_sum_lweights(theta, key, n_particles, y_meas, model):
    """
    
    Args:
        theta: A `jnp.array` that represents the values of the parameters.
        key: The key required for the prng.
        n_particles: The number of particles to use in the particle filter.
        y_meas: The measurements of the observations required for the particle filter.
        
    Returns:
        The sum of the particle log weights from the particle filters.
    """
    
    ret = particle_filter(model, y_meas, theta, n_particles, key)
    sum_particle_lweights = particle_loglik(ret['logw_particles'])
    return sum_particle_lweights


def stoch_opt(model, params, grad_fun, y_meas, n_particles=100, iterations=10, learning_rate=0.01, key=1):
    """
    
    Args:
        params: A `jnp.array` that represents the initial values of the parameters.
        grad_fun: The function which we would like to take the gradient with respect to.
        y_meas: The measurements of the observations required for the particle filter.
        n_particles: The number of particles to use in the particle filter.
        learning_rate: The learning rate for the gradient descent algorithm.
        iterations: The number of iterations to run the gradient descent for.
        key: The key required for the prng.
        
    Returns:
        The stochastic approximation of `theta` which are the parameters of the model. 
    """
    grad_lweights = jax.jit(jax.grad(grad_fun, argnums=0), static_argnums=(2,4))
    for i in range(iterations):
        key, subkey = random.split(key)
        params_update = grad_lweights(params, subkey, n_particles, y_meas, model)
        params = params + (learning_rate * params_update)
        
    return params