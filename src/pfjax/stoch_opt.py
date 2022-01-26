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


def update_params(params, subkey, grad_fun=None, n_particles=100, y_meas=None, model=None, learning_rate=0.01, mask=None, **kwargs):
    temp = 0 # grad_fun(params, subkey, n_particles, y_meas, model)   # Remove me if not debugging FIXME:
    params_update = jax.grad(grad_fun)(
        params, subkey, n_particles, y_meas, model, **kwargs)
    return (jnp.where(mask, params_update, 0)), temp


def stoch_opt(model, params, grad_fun, y_meas, n_particles=100, iterations=10,
              learning_rate=0.01, key=1, mask=None, **kwargs):
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
    partial_update_params = partial(update_params, n_particles=n_particles, y_meas=y_meas,
                                    model=model, learning_rate=learning_rate, mask=mask, grad_fun=grad_fun, **kwargs)
    update_fn = jax.jit(partial_update_params)
    gradients = []
    stoch_obj = []
    keys = random.split(key, iterations)
    for subkey in keys:
        update_vals, temp = update_fn(params, subkey)
        params = params + learning_rate * update_vals
        stoch_obj.append(temp)
        gradients.append(update_vals)
    return params, stoch_obj, gradients
