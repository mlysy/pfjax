"""
Particle filter using multivariate normal distribution in JAX.

Uses the same API as NumPy/SciPy version.

### Variant 1: 
- Removing the X_particles matrix because it makes no sense for MVN, also getting some issues with assignment of samples from MVN to matrix.
    Instead, going to leave X_particles as a single (n_particles, n_states) martix that gets updated every iteration 
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import lax
from jax.experimental.maps import xmap
from functools import partial



def particle_resample_mvn(particles, logw, key):
    """
    Approximate particle distribution with MVN. Uses weighted mean and covariance of `particles` for the MVN mean and cov

    Args:
        particles: Matrix of size (n_particles, n_states) containing particles at the current timestep
        logw: Vector of `n_particles` unnormalized log-weights
        key: PRNG key

    Returns: 
        Matrix of size (`n_particles`, `n_states`) representing the particles for the next timestep 
    """
    n_particles = particles.shape[0]
    n_states = particles.shape[1]

    wgt = jnp.exp(logw - jnp.max(logw))
    prob = wgt / jnp.sum(wgt)
    # variables are the rows in jnp.cov()
    cov_mat = jnp.cov(jnp.transpose(particles), aweights=prob)
    mu = jnp.average(particles, axis=0, weights=prob)

    if n_states == 1:
        # cant use MVNormal, Jax complains
        Z = random.normal(key, shape=(n_particles, ))
        samples = mu + jnp.sqrt(cov_mat) * Z
        samples = samples.reshape(n_particles, n_states)
    else : 
        samples = random.multivariate_normal(key, mean=mu,
                                        cov=cov_mat,
                                        shape=(n_particles, ))
        samples = samples.reshape(n_particles, n_states)
    return samples, mu, cov_mat

def particle_filter_for(model, y_meas, theta, n_particles, key):
    """
    Apply particle filter for given value of `theta`.

    Closely follows Algorithm 2 of https://arxiv.org/pdf/1306.3277.pdf.

    This is the testing version which does the following:

    - Uses for-loops instead of `lax.scan` and `vmap/xmap`.
    - Only performs a bootstrap particle filter using `state_sample()` and `meas_lpdf()`.

    **FIXME:** Refactor `ancestor_particles` to use particle_resample_mvn

    Args:
        model: Object specifying the state-space model.
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
    X_particles_mu = jnp.zeros((n_obs, model.n_state))
    X_particles = jnp.zeros((n_particles, model.n_state))
    logw_particles = jnp.zeros((n_obs, n_particles))
    ancestor_particles = jnp.zeros((n_obs, n_particles), dtype=int)
    # initial particles have no ancestors
    ancestor_particles = ancestor_particles.at[0].set(-1)
    # initial time point
    key, *subkeys = random.split(key, num=n_particles+1)
    for p in range(n_particles):
        X_particles = X_particles.at[p].set(
            model.init_sample(y_meas[0], theta, subkeys[p])
        )
        logw_particles = logw_particles.at[0, p].set(
            model.init_logw(X_particles[p], y_meas[0], theta)
        )
    # X_particles = X_particles.at[0].set(
    #     jax.vmap(lambda k: model.init_sample(y_meas[0], theta, k))(
    #         jnp.array(subkeys)
    #     )
    # )
    # logw_particles = logw_particles.at[0].set(
    #     jax.vmap(lambda xs: model.init_logw(xs, y_meas[0], theta) +
    #              model.meas_lpdf(y_meas[0], xs, theta))(X_particles[0])
    # )
    # subsequent time points
    for t in range(1, n_obs):
        # resampling step
        key, subkey = random.split(key)
        # ancestor_particles = ancestor_particles.at[t].set(
        #     particle_resample(logw_particles[t-1], subkey)
        # )
        resampled_particles = particle_resample_mvn(
            X_particles, logw_particles[t-1], subkey)
        X_particles = resampled_particles[0]
        X_particles_mu = X_particles_mu.at[t].set(resampled_particles[1]) # mean of MVN

        # print("Particles: \n")
        # print(X_particles)

        # update
        key, *subkeys = random.split(key, num=n_particles+1)
        for p in range(n_particles):
            X_particles = X_particles.at[p].set(model.state_sample(
                X_particles[p], theta, subkeys[p])
            )
            logw_particles = logw_particles.at[t, p].set(
                model.meas_lpdf(y_meas[t], X_particles[p], theta)
            )
        # X_particles = X_particles.at[t].set(
        #     jax.vmap(lambda xs, k: model.state_sample(xs, theta, k))(
        #         X_particles[t-1, ancestor_particles[t]], jnp.array(subkeys)
        #     )
        # )
        # logw_particles = logw_particles.at[t].set(
        #     jax.vmap(lambda xs: model.meas_lpdf(y_meas[t], xs, theta))(
        #         X_particles[t]
        #     )
        # )
    return {
        "X_particles": X_particles,
        "logw_particles": logw_particles,
        "ancestor_particles": ancestor_particles
    }


# @partial(jax.jit, static_argnums=2)
def particle_filter(model, y_meas, theta, n_particles, key):
    """
    Apply particle filter using a MVN approximation of the particle distribution for given value of `theta`.
    This model does not return `ancestor_particles`, as there is no ancestry for the MVN

    Closely follows Algorithm 2 of https://arxiv.org/pdf/1306.3277.pdf.

    Args:
        model: Object specifying the state-space model.
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        n_particles: Number of particles.
        key: PRNG key.

    Returns:
        A dictionary with elements:
            - `X_particles`: An `ndarray` with leading dimensions `(n_obs, n_particles)` containing the mean of the MVN at each timestep. Note that this is different from the vanilla particle filter, which returns the particles at each timestep
            - `logw_particles`: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.
    """
    n_obs = y_meas.shape[0]

    # lax.scan setup
    # scan function
    def fun(carry, t):
        # resampling step
        key, subkey = random.split(carry["key"])
        # ancestor_particles = particle_resample(carry["logw_particles"],
        #                                        subkey)
        resampled_particles = particle_resample_mvn(
            X_particles, carry["logw_particles"], subkey)
        X_particles = resampled_particles[0]
        X_particles_mu = X_particles_mu.at[t].set(
            resampled_particles[1])  # mean of MVN

        # update particles
        key, *subkeys = random.split(key, num=n_particles+1)
        X_particles, logw_particles = jax.vmap(
            lambda xs, k: model.pf_step(xs, y_meas[t], theta, k)
        )(carry["X_particles"][ancestor_particles], jnp.array(subkeys))

        # X_particles = jax.vmap(lambda xs, k: model.state_sample(xs, theta, k))(
        #     carry["X_particles"][ancestor_particles], jnp.array(subkeys)
        # )
        # # update log-weights
        # logw_particles = jax.vmap(
        #     lambda xs: model.meas_lpdf(y_meas[t], xs, theta)
        # )(X_particles)
        # breakpoint()
        # output
        res = {
            "logw_particles": logw_particles,
            "X_particles": X_particles,
            "key": key
        }
        return res, res
    # scan initial value
    key, *subkeys = random.split(key, num=n_particles+1)
    # vmap version
    X_particles, logw_particles = jax.vmap(
        lambda k: model.pf_init(y_meas[0], theta, k))(jnp.array(subkeys))
    # X_particles = jax.vmap(
    #     lambda k: model.init_sample(y_meas[0], theta, k))(jnp.array(subkeys))
    # logw_particles = jax.vmap(
    #     lambda xs: model.init_logw(xs, y_meas[0], theta))(X_particles)
    # xmap version: experimental!
    # X_particles = xmap(
    #     lambda ym, th, k: model.init_sample(ym, th, k),
    #     in_axes=([...], [...], ["particles", ...]),
    #     out_axes=["particles", ...])(y_meas[0], theta, jnp.array(subkeys))
    # logw_particles = xmap(
    #     lambda xs, ym, th: model.init_logw(xs, ym, th),
    #     in_axes=(["particles", ...], [...], [...]),
    #     out_axes=["particles", ...])(X_particles, y_meas[0], theta)
    init = {
        "X_particles": X_particles,
        "logw_particles": logw_particles,
        "key": key
    }
    # lax.scan itself
    last, full = lax.scan(fun, init, jnp.arange(1, n_obs))
    # append initial values
    out = {
        k: jnp.append(jnp.expand_dims(init[k], axis=0), full[k], axis=0)
        for k in ["X_particles", "logw_particles"]
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
