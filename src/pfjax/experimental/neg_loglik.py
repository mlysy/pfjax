import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import lax


def particle_neg_loglik(theta, key, n_particles, y_meas, model):
    """
    Evaluate the bootstrap particle filter estimate of the negative log-likelihood at parameter values \theta. Runs the particle filter for each timestep in y_meas and sums the log-weights for each particle

    Args:
        theta: A `jnp.array` that represents the values of the parameters.
        key: The key required for the prng.
        n_particles: The number of particles to use in the particle filter.
        y_meas: The measurements of the observations required for the particle filter.

    Returns:
        Estimate of the negative log-likelihood evaluated at \theta.
    """
    ret = particle_filter(model, key, y_meas, theta, n_particles)
    sum_particle_lweights = particle_loglik(ret['logw'])
    return -sum_particle_lweights


def particle_neg_loglik_mvn(theta, key, n_particles, y_meas, model):
    """
    Evaluate the MVN particle filter estimate of the negative log-likelihood at parameter values \theta. Runs the particle filter for each timestep in y_meas and sums the log-weights for each particle

    Args:
        theta: A `jnp.array` that represents the values of the parameters.
        key: The key required for the prng.
        n_particles: The number of particles to use in the particle filter.
        y_meas: The measurements of the observations required for the particle filter.

    Returns:
        Estimate of the negative log-likelihood evaluated at \theta.
    """
    ret = particle_filter(model, key, y_meas, theta,
                          n_particles, particle_sampler=particle_resample_mvn)
    sum_particle_lweights = particle_loglik(ret['logw'])
    return -sum_particle_lweights
