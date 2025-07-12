import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
import ott
from jax import lax, random
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from pfjax.utils import logw_to_prob, tree_array2d, tree_shuffle


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
    ancestors = random.choice(
        key, a=jnp.arange(n_particles), shape=(n_particles,), p=prob
    )
    return {
        "x_particles": tree_shuffle(x_particles_prev, index=ancestors),
        "ancestors": ancestors,
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
    x_particles, unravel_fn = tree_array2d(x_particles_prev, shape0=n_particles)
    # x_particles = jnp.transpose(x_particles)
    # calculate weighted mean and variance
    mvn_mean = jnp.average(x_particles, axis=0, weights=prob)
    mvn_cov = jnp.atleast_2d(jnp.cov(x_particles, rowvar=False, aweights=prob))
    # for numeric stability
    mvn_cov += jnp.diag(jnp.ones(mvn_cov.shape[0]) * 1e-10)
    x_particles = random.multivariate_normal(
        key, mean=mvn_mean, cov=mvn_cov, shape=(n_particles,), method="eigh"
    )
    return {
        "x_particles": unravel_fn(x_particles),
        "mvn_mean": mvn_mean,
        "mvn_cov": mvn_cov,
    }


def resample_ot(
    key, x_particles_prev, logw, scaled=True, pointcloud_kwargs={}, sinkhorn_kwargs={}
):
    r"""
    Particle resampler using optimal transport.

    Based on Algorithms 2 and 3 of Corenflos et al 2021 <https://arxiv.org/abs/2102.07850>.

    **Notes:**

    - Argument `jit` to `ott.solvers.linear.sinkhorn.sinkhorn()` is ignored, i.e., always set to `False`.

    - Both `sinkhorn_kwargs` and `pointcloud_kwargs` are shallow copied inside the function to impure function effects.

    Args:
        key: PRNG key.
        x_particles_prev: An `ndarray` with leading dimension `n_particles` consisting of the particles from the previous time step.
        logw: Vector of corresponding `n_particles` unnormalized log-weights.
        scaled: Whether or not to divide `x_particles_prev` by `sqrt(x_particles_prev.shape[1]) * max(std(x_particles_prev, axis=0))`, after reshaping `x_particles` into a 2D array with leading dimension of size `n_particles`.  If `True` overrides any value of `pointcloud_kwargs["scale_cost"]`.
        pointcloud_kwargs: Dictionary of additional arguments to `ott.pointcloud.PointCloud()`.
        sinkhorn_kwargs: Dictionary of additional arguments to `ott.solvers.linear.sinkhorn.Sinkhorn()`.

    Returns:
        A dictionary with elements:
            - `x_particles`: An `ndarray` with leading dimension `n_particles` consisting of the particles from the current time step.
            - `sink`: An object of type `ott.solvers.linear.sinkhorn.SinkhornOutput`, as returned by `ott.solvers.linear.sinkhorn.Sinkhorn()`.
    """
    sinkhorn_kwargs = sinkhorn_kwargs.copy()
    pointcloud_kwargs = pointcloud_kwargs.copy()
    # sinkhorn_kwargs.update(jit=False) # depreciated argument
    prob = logw_to_prob(logw)
    # p_shape = x_particles_prev.shape
    # n_particles = p_shape[0]
    # x_particles = x_particles_prev.reshape((n_particles, -1))
    n_particles = logw.shape[0]
    x_particles, unravel_fn = tree_array2d(x_particles_prev, shape0=n_particles)
    if scaled:
        # can't jit compile pointcloud_kwargs.update(scale_cost=scale_cost)
        # this way.  So instead scale particles directly.
        scale_cost = jnp.max(jnp.var(x_particles, axis=0))
        scale_cost = jnp.sqrt(x_particles.shape[1] * scale_cost)
        x_particles_scaled = x_particles / scale_cost
        pointcloud_kwargs.update(scale_cost=1.0)
    else:
        x_particles_scaled = x_particles
    geom = pointcloud.PointCloud(
        x=x_particles_scaled, y=x_particles_scaled, **pointcloud_kwargs
    )
    problem = linear_problem.LinearProblem(
        geom, a=jnp.ones(n_particles) / n_particles, b=prob
    )
    solver = sinkhorn.Sinkhorn(**sinkhorn_kwargs)
    sink = solver(problem)
    # sink = sinkhorn.sinkhorn(geom,
    #                          a=prob,
    #                          b=jnp.ones(n_particles)/n_particles,
    #                          **sinkhorn_kwargs)
    x_particles = n_particles * sink.apply(inputs=x_particles.T, axis=1)
    return {"x_particles": unravel_fn(x_particles.T), "sink": sink}
