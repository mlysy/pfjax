"""
Complete data loglikelihood for a state space model.

The API requires the user to define a model class with the following methods:

- `state_lpdf()`
- `meas_lpdf()`

The provided function is:

- `loglik_full()`
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from jax import random
from jax import lax
from .utils import *


def loglik_full(model, y_meas, x_state, theta):
    """
    Calculate the joint loglikelihood `p(y_{0:T} | x_{0:T}, theta) * p(x_{0:T} | theta)`.

    Args:
        model: Object specifying the state-space model.
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        x_state: The sequence of `n_obs` state variables `x_state = (x_0, ..., x_T)`.
        theta: Parameter value.

    Returns:
        The value of the loglikelihood.
    """
    n_obs = y_meas.shape[0]
    # initial measurement
    ll_init = model.meas_lpdf(y_curr=y_meas[0], x_curr=x_state[0],
                              theta=theta)
    # subsequent measurements and state variables
    ll_step = jax.vmap(
        lambda xc, xp, yc:
        model.state_lpdf(x_curr=xc, x_prev=xp, theta=theta) +
        model.meas_lpdf(y_curr=yc, x_curr=xc, theta=theta)
    )(tree_rm_first(x_state), tree_rm_last(x_state), tree_rm_first(y_meas))
    # ll_step = jax.vmap(lambda t:
    #                    model.state_lpdf(x_curr=x_state[t],
    #                                     x_prev=x_state[t-1],
    #                                     theta=theta) +
    #                    model.meas_lpdf(y_curr=y_meas[t],
    #                                    x_curr=x_state[t],
    #                                    theta=theta))(jnp.arange(1, n_obs))
    return ll_init + jnp.sum(ll_step)
