import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from jax import random
from jax import lax
# from jax.experimental.host_callback import id_print
import ott
from ott.geometry import pointcloud
from ott.core import sinkhorn
from .utils import lwgt_to_prob, marginal_cdf, weighted_corr, sort_marginal, continuous_cdf


# @partial(jax.jit, static_argnums=(3,))
def resample_gaussian_copula(key, x_particles_prev, logw):
    r"""
    Particle resampler with Gaussian Copula distribution.

    Estimate and sample from a Gaussian copula as follows: 
        - Find Y = (F_1(X_1), ..., F_d(X_d))
        - rho_hat = weighted correlation of Y 
        - Sample Z ~ N(0, rho_hat) N times
        - Create U = (phi(Z_1), ..., phi(Z_d))
        - Use inverse-CDF of marginals to create samples: (inv-CDF(U_1), ..., inv-CDF(U_d))

    Args: 
        key: PRNG key
        x_particles_prev: An `ndarray` with leading dimension `n_particles` consisting of the particles from the previous time step.
        logw: Vector of corresponding `n_particles` unnormalized log-weights.

    Returns: 
        A dictionary with elements:
            - `x_particles`: An `ndarray` with leading dimension `n_particles` consisting of the particles from the current time step.  These are sampled with replacement from `x_particles_prev` with probability vector `exp(logw) / sum(exp(logw))`.
            - `ancestors`: Vector of `n_particles` integers between 0 and `n_particles-1` giving the index of each element of `x_particles_prev` corresponding to the elements of `x_particles`.

    TODO: 
        - Change `continuous_cdf` to accept a vector of U
        - See if we can optimize any other parts of the resampler to reduce the number of vmaps
    """
    N, d = x_particles_prev.shape
    prob = lwgt_to_prob(logw)

    # estimate correlation matrix: 
    Y = jax.vmap(
        lambda x: 
        jax.vmap(
            marginal_cdf,
            in_axes = (0, None, None)
        )(x, x, prob),
        in_axes = (1))(x_particles_prev)
    rho_hat = weighted_corr(Y, weights = prob)

    # gaussian copula: 
    Z = random.multivariate_normal(key=key, mean = jnp.zeros(d), cov = rho_hat, shape = (N,))
    U = jax.scipy.stats.norm.cdf(Z)

    sorted_marginals = jax.vmap(
        sort_marginal,
        in_axes = (1, None)
    )(x_particles_prev, prob)

    x_samples = jax.vmap(
        lambda x, w, u: jax.vmap(
            continuous_cdf,
            in_axes= (None, None, 0)
        )(x, w, u),
        in_axes = (0, 0, 1))(sorted_marginals["x"], sorted_marginals["w"], U)

    return {
        "x_particles": x_samples
    }


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
