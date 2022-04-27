"""
Particle filter in JAX.
The API requires the user to define a model class with the following methods:
- `pf_init()`
- `pf_step()`
The provided functions are:
- `particle_filter()`
- `particle_loglik()`
- `particle_smooth()`
- `particle_resample()`
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from jax import random
from jax import lax
# from jax.experimental.maps import xmap
import ott
from ott.geometry import pointcloud
from ott.core import sinkhorn


def lwgt_to_prob(logw):
    r"""
    Calculate normalized probabilities from unnormalized log weights.
    Args:
        logw: Vector of `n_particles` unnormalized log-weights.
    Returns:
        Vector of `n_particles` normalized weights that sum to 1.
    """
    wgt = jnp.exp(logw - jnp.max(logw))
    prob = wgt / jnp.sum(wgt)
    return prob


def resample_multinomial(key, x_particles_prev, logw):
    r"""
    Particle resampler.
    This basic one just does a multinomial sampler, i.e., sample with replacement proportional to weights.
    Args:
        key: PRNG key.
        x_particles_prev: An `ndarray` with leading dimension `n_particles` consisting of the particles from the previous time step.
        logw: Vector of corresponding `n_particles` unnormalized log-weights.
    Returns:
        A dictionary with elements:
            - `x_particles`: An `ndarray` with leading dimension `n_particles` consisting of the particles from the current time step.  These are sampled with replacement from `x_particles_prev` with probability vector `exp(logw) / sum(exp(logw))`.
            - `ancestors`: Vector of `n_particles` integers between 0 and `n_particles-1` giving the index of each element of `x_particles_prev` corresponding to the elements of `x_particles`.
    """
    prob = lwgt_to_prob(logw)
    n_particles = logw.size
    ancestors = random.choice(key,
                              a=jnp.arange(n_particles),
                              shape=(n_particles,), p=prob)
    return {
        "x_particles": x_particles_prev[ancestors, ...],
        "ancestors": ancestors
    }


def resample_mvn(key, x_particles_prev, logw):
    r"""
    Particle resampler with Multivariate Normal approximation.
    Args:
        key: PRNG key.
        x_particles_prev: An `ndarray` with leading dimension `n_particles` consisting of the particles from the previous time step.
        logw: Vector of corresponding `n_particles` unnormalized log-weights.
    Returns:
        A dictionary with elements:
            - `x_particles`: An `ndarray` with leading dimension `n_particles` consisting of the particles from the current time step.
            - `mvn_mean`: Vector of length `n_state = prod(x_particles.shape[1:])` representing the mean of the MVN.
            - `mvn_cov`: Matrix of size `n_state x n_state` representing the covariance matrix of the MVN.
    """
    prob = lwgt_to_prob(logw)
    p_shape = x_particles_prev.shape
    n_particles = p_shape[0]
    # calculate weighted mean and variance
    x_particles = jnp.transpose(x_particles_prev.reshape((n_particles, -1)))
    mvn_mean = jnp.average(x_particles, axis=1, weights=prob)
    mvn_cov = jnp.atleast_2d(jnp.cov(x_particles, aweights=prob))
    # for numeric stability
    mvn_cov += jnp.diag(jnp.ones(mvn_cov.shape[0]) * 1e-10)
    x_particles = random.multivariate_normal(key,
                                             mean=mvn_mean,
                                             cov=mvn_cov,
                                             shape=(n_particles,))
    return {
        "x_particles": jnp.reshape(x_particles, newshape=p_shape),
        "mvn_mean": mvn_mean,
        "mvn_cov": mvn_cov
    }


def resample_ot(key, x_particles_prev, logw,
                pointcloud_kwargs={},
                sinkhorn_kwargs={}):
    r"""
    Particle resampler using optimal transport.
    Based on Algorithms 2 and 3 of Corenflos et al 2021 <https://arxiv.org/abs/2102.07850>.
    **Notes:**
    - Argument `jit` to `ott.sinkhorn.sinkhorn()` is ignored, i.e., always set to `False`.
    Args:
        key: PRNG key.
        x_particles_prev: An `ndarray` with leading dimension `n_particles` consisting of the particles from the previous time step.
        logw: Vector of corresponding `n_particles` unnormalized log-weights.
        pointcloud_kwargs: Dictionary of additional arguments to `ott.pointcloud.PointCloud()`.
        sinkhorn_kwargs: Dictionary of additional arguments to `ott.sinkhorn.sinkhorn()`.
    Returns:
        A dictionary with elements:
            - `x_particles`: An `ndarray` with leading dimension `n_particles` consisting of the particles from the current time step.
            - `geom`: An `ott.Geometry` object.
            - `sink`: The output of the call to `ott.sinkhorn.sinkhorn()`.
    """
    sinkhorn_kwargs.update(jit=False)
    prob = lwgt_to_prob(logw)
    p_shape = x_particles_prev.shape
    n_particles = p_shape[0]
    x_particles = x_particles_prev.reshape((n_particles, -1))
    geom = pointcloud.PointCloud(x=x_particles, y=x_particles,
                                 **pointcloud_kwargs)
    sink = sinkhorn.sinkhorn(geom,
                             a=prob,
                             b=jnp.ones(n_particles),
                             **sinkhorn_kwargs)
    x_particles = geom.apply_transport_from_potentials(
        f=sink.f, g=sink.g, vec=x_particles.T
    )
    return {
        "x_particles": jnp.reshape(x_particles.T, newshape=p_shape),
        "geom": geom,
        "sink": sink
    }


def particle_filter(model, key, y_meas, theta, n_particles,
                    resampler=resample_multinomial):
    r"""
    Apply particle filter for given value of `theta`.
    Closely follows Algorithm 2 of Murray 2013 <https://arxiv.org/abs/1306.3277>.
    Args:
        model: Object specifying the state-space model.
        key: PRNG key.
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        n_particles: Number of particles.
        resampler: Function used at step `t` to obtain sample of particles from `p(x_{t-1} | y_{0:t-1}, theta)`.  The inputs to the function are `resampler(x_particles, logw, key)`, and the return value is a dictionary with mandatory element `x_particles` and optional elements that get stacked to the final output using `lax.scan()`.  Default value is `resample_multinomial()`.
    Returns:
        A dictionary with elements:
            - `x_particles`: An `ndarray` with leading dimensions `(n_obs, n_particles)` containing the state variable particles.
            - `logw`: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.
            - `...`: Other `ndarray`s with leading dimension `n_obs-1`, corresponding to additional outputs from `resampler()` as accumulated by `lax.scan()`.  Since these additional outputs do not apply to the first time step (since it has no previous time step), the leading dimension of each additional output is `n_obs-1`.
    """
    n_obs = y_meas.shape[0]

    # lax.scan setup
    # scan function
    def fun(carry, t):
        # sample particles from previous time point
        key, subkey = random.split(carry["key"])
        new_particles = resampler(
            key=subkey,
            x_particles_prev=carry["x_particles"],
            logw=carry["logw"]
        )
        # update particles to current time point (and get weights)
        key, *subkeys = random.split(key, num=n_particles+1)
        x_particles, logw = jax.vmap(
            lambda xs, k: model.pf_step(k, xs, y_meas[t], theta)
        )(new_particles["x_particles"], jnp.array(subkeys))
        # output
        res_carry = {
            "x_particles": x_particles,
            "logw": logw,
            "key": key
        }
        res_stack = new_particles
        res_stack["x_particles"] = x_particles
        res_stack["logw"] = logw
        return res_carry, res_stack
    # scan initial value
    key, *subkeys = random.split(key, num=n_particles+1)
    # vmap version
    x_particles, logw = jax.vmap(
        lambda k: model.pf_init(k, y_meas[0], theta))(jnp.array(subkeys))
    # xmap version: experimental!
    # x_particles = xmap(
    #     lambda ym, th, k: model.init_sample(ym, th, k),
    #     in_axes=([...], [...], ["particles", ...]),
    #     out_axes=["particles", ...])(y_meas[0], theta, jnp.array(subkeys))
    # logw = xmap(
    #     lambda xs, ym, th: model.init_logw(xs, ym, th),
    #     in_axes=(["particles", ...], [...], [...]),
    #     out_axes=["particles", ...])(x_particles, y_meas[0], theta)
    init = {
        "x_particles": x_particles,
        "logw": logw,
        "key": key
    }
    resampler(key=init["key"], x_particles_prev=init["x_particles"],
              logw=init["logw"])
    # lax.scan itself
    last, full = lax.scan(fun, init, jnp.arange(1, n_obs))
    # append initial values of x_particles and logw
    full["x_particles"] = jnp.append(
        jnp.expand_dims(init["x_particles"], axis=0),
        full["x_particles"], axis=0)
    full["logw"] = jnp.append(
        jnp.expand_dims(init["logw"], axis=0),
        full["logw"], axis=0)
    return full


def particle_loglik(logw):
    r"""
    Calculate particle filter marginal loglikelihood.
    Args:
        logw: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.
    Returns:
        Particle filter approximation of
        ```
        log p(y_meas | theta) = log int p(y_meas | x_state, theta) * p(x_state | theta) dx_state
        ```
    """
    n_particles = logw.shape[1]
    return jnp.sum(jsp.special.logsumexp(logw, axis=1) - jnp.log(n_particles))


def particle_smooth(key, logw, x_particles, ancestors):
    r"""
    Draw a sample from `p(x_state | x_meas, theta)` using the basic particle smoothing algorithm.
    **FIXME:**
    - Will probably need to change inputs to "generalize" to other resampling methods.
    Args:
        key: PRNG key.
        logw: Vector of `n_particles` unnormalized log-weights at the last time point `t = n_obs-1`.
<<<<<<< HEAD
        x_particles: An `ndarray` with leading dimensions `(n_obs, n_particles)` containing the state variable particles.
        ancestors: An integer `ndarray` of shape `(n_obs, n_particles)` where each element gives the index of the particle's ancestor at the previous time point.
=======
        x_particles: JAX array with leading dimensions `(n_obs, n_particles)` containing the state variable particles.
        ancestors: JAX integer array of shape `(n_obs-1, n_particles)` where each element gives the index of the particle's ancestor at the previous time point.

>>>>>>> 143624d694c0216bf486ca4dba98588d822f38fc
    Returns:
        An `ndarray` with leading dimension `n_obs` sampled from `p(x_{0:T} | y_{0:T}, theta)`.
    """
    n_particles = logw.size
    n_obs = x_particles.shape[0]
    prob = lwgt_to_prob(logw)
    # wgt = jnp.exp(logw - jnp.max(logw))
    # prob = wgt / jnp.sum(wgt)

    # lax.scan setup
    # scan function
    def fun(carry, t):
        # ancestor particle index
        i_part = ancestors[t, carry["i_part"]]
        res = {"i_part": i_part}
        return res, res
        # res_carry = {"i_part": i_part}
        # res_stack = {"i_part": i_part, "x_state": x_particles[t, i_part]}
        # return res_carry, res_stack
    # scan initial value
    init = {
        "i_part": random.choice(key, a=jnp.arange(n_particles), p=prob)
    }
    # lax.scan itself
    last, full = lax.scan(fun, init, jnp.flip(jnp.arange(n_obs-1)))
    # particle indices in forward order
    i_part = jnp.flip(jnp.append(init["i_part"], full["i_part"]))
    return x_particles[jnp.arange(n_obs), i_part, ...]  # , i_part