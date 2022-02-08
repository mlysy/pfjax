"""
Stochastic optimization for particle filter.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from jax import random
from jax import lax
from jax.experimental.maps import xmap
from functools import partial
from pfjax import particle_loglik, particle_filter, particle_resample_mvn


def get_sum_lweights(theta, key, n_particles, y_meas, model, **kwargs):
    """

    Args:
        theta: A `jnp.array` that represents the values of the parameters.
        key: The key required for the prng.
        n_particles: The number of particles to use in the particle filter.
        y_meas: The measurements of the observations required for the particle filter.

    Returns:
        The sum of the particle log weights from the particle filters.
    """
    ret = particle_filter(model, key, y_meas, theta, n_particles)
    sum_particle_lweights = particle_loglik(ret['logw'])
    return sum_particle_lweights


def get_sum_lweights_mvn(theta, key, n_particles, y_meas, model):
    """
    FIXME: plz delete me, only for testing 
    """
    # model, key, y_meas, theta, n_particles,particle_sampler=particle_resample):
    ret = particle_filter(model = model, y_meas = y_meas, theta = theta, n_particles = n_particles,
                          key = key, particle_sampler=particle_resample_mvn)
    sum_particle_lweights = particle_loglik(ret['logw'])
    return sum_particle_lweights


def update_params(params, subkey, opt_state, grad_fun=None, n_particles=100, y_meas=None, model=None, learning_rate=0.01, mask=None,
                  optimizer=None, **kwargs):
    params_update = jax.grad(grad_fun, argnums=0)(
        params, subkey, n_particles, y_meas, model)
    params_update = jnp.where(mask, params_update, 0)
    updates, opt_state = optimizer.update(params_update, opt_state)
    return optax.apply_updates(params, updates)


def stoch_opt(model, params, grad_fun, y_meas, n_particles=100, iterations=10,
              learning_rate=0.01, key=1, mask=None):
    """
    Args:
        model: The model class for which all of the functions are defined.
        params: A jnp.array that represents the initial values of the parameters.
        grad_fun: The function which we would like to take the gradient with respect to.
        y_meas: The measurements of the observations required for the particle filter.
        n_particles: The number of particles to use in the particle filter.
        iterations: The number of iterations to run the gradient descent for.
        learning_rate: The learning rate for the gradient descent algorithm.
        key: The key required for the prng.
        mask: The mask over which dimensions we would like to perform the optimization.
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    partial_update_params = partial(update_params, n_particles=n_particles, y_meas=y_meas,
                                    model=model, learning_rate=learning_rate, mask=mask, grad_fun=grad_fun, optimizer=optimizer)
    update_fn = jax.jit(partial_update_params, donate_argnums=(0,))
    keys = random.split(key, iterations)
    for subkey in keys:
        params = update_fn(params, subkey, opt_state)
    return params
