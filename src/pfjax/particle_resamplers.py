import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from jax import random
from jax import lax
import ott
from ott.geometry import pointcloud
from ott.solvers.linear import sinkhorn
from .utils import logw_to_prob, tree_array2d


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
    prob = logw_to_prob(logw)
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
    prob = logw_to_prob(logw)
    # convert particles to 2d array
    # p_shape = x_particles_prev.shape
    # n_particles = p_shape[0]
    # x_particles = jnp.transpose(x_particles_prev.reshape((n_particles, -1)))
    n_particles = logw.shape[0]
    x_particles, unravel_fn = tree_array2d(x_particles_prev,
                                           shape0=n_particles)
    x_particles = jnp.transpose(x_particles)
    # calculate weighted mean and variance
    mvn_mean = jnp.average(x_particles, axis=1, weights=prob)
    mvn_cov = jnp.atleast_2d(jnp.cov(x_particles, aweights=prob))
    # for numeric stability
    mvn_cov += jnp.diag(jnp.ones(mvn_cov.shape[0]) * 1e-10)
    x_particles = random.multivariate_normal(key,
                                             mean=mvn_mean,
                                             cov=mvn_cov,
                                             shape=(n_particles,))
    return {
        "x_particles": unravel_fn(x_particles),
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

    - Argument `jit` to `ott.solvers.linear.sinkhorn.sinkhorn()` is ignored, i.e., always set to `False`.

    - Can't seem to jit with all types of `scale_cost` passed directly to  `pointcloud.PointCloud()`; `float`s seem to work but not `jax.numpy.array(float)`.

    Args:
        key: PRNG key.
        x_particles_prev: An `ndarray` with leading dimension `n_particles` consisting of the particles from the previous time step.
        logw: Vector of corresponding `n_particles` unnormalized log-weights.
        pointcloud_kwargs: Dictionary of additional arguments to `ott.pointcloud.PointCloud()`.
        sinkhorn_kwargs: Dictionary of additional arguments to `ott.solvers.linear.sinkhorn.sinkhorn()`.

    Returns:
        A dictionary with elements:
            - `x_particles`: An `ndarray` with leading dimension `n_particles` consisting of the particles from the current time step.
            - `sink`: An object of type `ott.solvers.linear.sinkhorn.SinkhornOutput`, as returned by `ott.solvers.linear.sinkhorn.sinkhorn()`.
    """
    sinkhorn_kwargs.update(jit=False)
    prob = logw_to_prob(logw)
    # p_shape = x_particles_prev.shape
    # n_particles = p_shape[0]
    # x_particles = x_particles_prev.reshape((n_particles, -1))
    n_particles = logw.shape[0]
    x_particles, unravel_fn = tree_array2d(x_particles_prev,
                                           shape0=n_particles)
    # x_particles = jnp.transpose(x_particles)
    # if "scale_cost" not in pointcloud_kwargs:
    #     scale_cost = jnp.max(jnp.std(x_particles, axis=0))
    #     scale_cost = jnp.sqrt(x_particles.shape[1]) * scale_cost
    #     x_particles_scaled = x_particles/scale_cost
    # else:
    #     x_particles_scaled = x_particles
    geom = pointcloud.PointCloud(x=x_particles,
                                 y=x_particles,
                                 **pointcloud_kwargs)
    sink = sinkhorn.sinkhorn(geom,
                             a=prob,
                             b=jnp.ones(n_particles)/n_particles,
                             **sinkhorn_kwargs)
    x_particles = sink.apply(inputs=x_particles.T, axis=1)
    return {
        "x_particles": unravel_fn(x_particles.T),
        "sink": sink
    }
