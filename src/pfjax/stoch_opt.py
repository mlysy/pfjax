"""
Stochastic optimization for particle filter.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import lax
from jax.experimental.maps import xmap
from functools import partial
from pfjax import particle_loglik, particle_filter


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


def update_params(params, subkey, grad_fun=None, n_particles=100, y_meas=None, model=None, learning_rate=0.01, mask=None):
    params_update = jax.grad(grad_fun, argnums=0)(
        params, subkey, n_particles, y_meas, model)
    return params + learning_rate * (jnp.where(mask, params_update, 0))


def stoch_opt(model, params, grad_fun, y_meas, n_particles=100, iterations=10,
              learning_rate=0.01, key=1, mask=None):
    """
    Args:
        params: A jnp.array that represents the initial values of the parameters.
        grad_fun: The function which we would like to take the gradient with respect to.
        y_meas: The measurements of the observations required for the particle filter.
        n_particles: The number of particles to use in the particle filter.
        learning_rate: The learning rate for the gradient descent algorithm.
        iterations: The number of iterations to run the gradient descent for.
        key: The key required for the prng.

    Returns:
        The stochastic approximation of theta which are the parameters of the model.
    """
    partial_update_params = partial(update_params, n_particles=n_particles, y_meas=y_meas,
                                    model=model, learning_rate=learning_rate, mask=mask, grad_fun=grad_fun)
    update_fn = jax.jit(partial_update_params, donate_argnums=(0,))
    keys = random.split(key, iterations)
    for subkey in keys:
        params = update_fn(params, subkey)
        print(params)
    return params
