import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from jax import random
from jax import lax
from pfjax.utils import *


def loglik_full(model, y_meas, x_state, theta):
    """
    Calculate the complete data loglikelihood for a state space model.

    Calculates `p(y_{0:T} | x_{0:T}, theta) * p(x_{0:T} | theta)`.

    Args:
        model: Object specifying the state-space model having the following methods:

            - `state_lpdf : (x_curr, x_prev, theta) -> lpdf`: Calculates the log-density of the state model.

            - `meas_lpdf : (y_curr, x_curr, theta) -> lpdf`: Calculates the log-density of the measurement model.

        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        x_state: The sequence of `n_obs` state variables `x_state = (x_0, ..., x_T)`.
        theta: Parameter value.

    Returns:
        The value of the complete data loglikelihood.
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
    )(tree_remove_first(x_state), tree_remove_last(x_state), tree_remove_first(y_meas))
    # ll_step = jax.vmap(lambda t:
    #                    model.state_lpdf(x_curr=x_state[t],
    #                                     x_prev=x_state[t-1],
    #                                     theta=theta) +
    #                    model.meas_lpdf(y_curr=y_meas[t],
    #                                    x_curr=x_state[t],
    #                                    theta=theta))(jnp.arange(1, n_obs))
    return ll_init + jnp.sum(ll_step)
