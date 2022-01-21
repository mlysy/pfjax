"""
MCMC algorithms for state space models.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import lax
from jax.experimental.maps import xmap
from functools import partial


class FlatPrior:
    """
    Flat prior class.
    """

    def lpdf(self, theta):
        return jnp.array(0.)


def full_loglik_for(model, y_meas, x_state, theta):
    """
    Calculate the joint loglikelihood `p(y_{0:T} | x_{0:T}, theta) * p(x_{0:T} | theta)`.

    For-loop version for testing.

    Args:
        model: Object specifying the state-space model.
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        x_state: The sequence of `n_obs` state variables `x_state = (x_0, ..., x_T)`.
        theta: Parameter value.

    Returns:
        The value of the loglikelihood.
    """
    n_obs = y_meas.shape[0]
    loglik = model.meas_lpdf(y_curr=y_meas[0], x_curr=x_state[0],
                             theta=theta)
    for t in range(1, n_obs):
        loglik = loglik + \
            model.state_lpdf(x_curr=x_state[t], x_prev=x_state[t-1],
                             theta=theta)
        loglik = loglik + \
            model.meas_lpdf(y_curr=y_meas[t], x_curr=x_state[t],
                            theta=theta)
    return loglik


def full_loglik(model, y_meas, x_state, theta):
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
    ll_step = jax.vmap(lambda t:
                       model.state_lpdf(x_curr=x_state[t],
                                        x_prev=x_state[t-1],
                                        theta=theta) +
                       model.meas_lpdf(y_curr=y_meas[t],
                                       x_curr=x_state[t],
                                       theta=theta))(jnp.arange(1, n_obs))
    return ll_init + jnp.sum(ll_step)


def param_mwg_update_for(model, prior, theta, x_state, y_meas, rw_sd, key):
    """
    Parameter update by Metropolis-within-Gibbs random walk.

    FIXME:
        - Assumes the parameters are real valued.  Next step would be to provide a validator to the model.
        - Gets size of `theta` from `theta` itself, rather than e.g., `model.n_param`.  
        - Potentially wastes an initial evaluation of `full_loglik(theta)`.  Could be passed in from a previous calculation but a bit cumbersome.

    Args:
        model: Object specifying the state-space model.
        prior: Object specifying the parameter prior.
        theta: Current parameter vector.
        x_state: The sequence of `n_obs` state variables `x_state = (x_0, ..., x_T)`.
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        rw_sd: Vector of length `n_param = theta.size` standard deviations for the componentwise random walk proposal.
        key: PRNG key.

    Returns:
        theta_update: Updated parameter vector.
        accept: Whether or not the proposal was accepted for each component: a vector of length `n_param`. 
    """
    n_param = theta.size
    theta_curr = theta + 0.  # how else to copy...
    accept = jnp.empty(n_param, dtype=bool)
    lp_curr = full_loglik(model, y_meas, x_state,
                          theta_curr) + prior.lpdf(theta_curr)
    for i in range(n_param):
        # 2 subkeys for each param: rw_jump and mh_accept
        key, *subkeys = random.split(key, num=3)
        # proposal
        theta_prop = theta_curr.at[i].set(
            theta_curr[i] + rw_sd[i] * random.normal(key=subkeys[0])
        )
        # acceptance rate
        lp_prop = full_loglik(model, y_meas, x_state,
                              theta_prop) + prior.lpdf(theta_prop)
        lrate = lp_prop - lp_curr
        # update parameter draw
        acc = random.bernoulli(key=subkeys[1],
                               p=jnp.maximum(1.0, jnp.exp(lrate)))
        theta_curr = theta_prop * acc
        lp_curr = lp_prop * acc
        accept = accept.at[i].set(acc)
    theta_update = theta_curr
    return theta_update, accept


def param_mwg_update(model, prior, theta, x_state, y_meas, rw_sd, key):
    """
    Parameter update by Metropolis-within-Gibbs random walk.

    FIXME:
        - Assumes the parameters are real valued.  Next step would be to provide a validator to the model.
        - Gets size of `theta` from `theta` itself, rather than e.g., `model.n_param`.  
        - Potentially wastes an initial evaluation of `full_loglik(theta)`.  Could be passed in from a previous calculation but a bit cumbersome.

    Args:
        model: Object specifying the state-space model.
        prior: Object specifying the parameter prior.
        theta: Current parameter vector.
        x_state: The sequence of `n_obs` state variables `x_state = (x_0, ..., x_T)`.
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        rw_sd: Vector of length `n_param = theta.size` standard deviations for the componentwise random walk proposal.
        key: PRNG key.

    Returns:
        theta_update: Updated parameter vector.
        accept: Whether or not the proposal was accepted for each component: a vector of length `n_param`. 
    """
    # lax.scan setup
    def fun(carry, i):
        lp_curr = carry["lp_curr"]
        theta_curr = carry["theta_curr"]
        key = carry["key"]
        # 2 subkeys for each param: rw_jump and mh_accept
        key, *subkeys = random.split(key, num=3)
        # proposal
        theta_prop = theta_curr.at[i].set(
            theta_curr[i] + rw_sd[i] * random.normal(key=subkeys[0])
        )
        # acceptance rate
        lp_prop = full_loglik(model, y_meas, x_state,
                              theta_prop) + prior.lpdf(theta_prop)
        lrate = lp_prop - lp_curr
        # update parameter draw
        accept = random.bernoulli(key=subkeys[1],
                                  p=jnp.maximum(1.0, jnp.exp(lrate)))
        res = {
            "theta_curr": theta_prop * accept,
            "lp_curr": lp_prop * accept,
            "accept": accept,
            "key": key
        }
        return res, res
    # scan initial value
    n_param = theta.size
    init = {
        "theta_curr": theta,
        "lp_curr": full_loglik(model, y_meas, x_state,
                               theta) + prior.lpdf(theta),
        "accept": jnp.array(True),
        "key": key
    }
    # scan itself
    last, full = lax.scan(fun, init, jnp.arange(n_param))
    return last["theta_curr"], full["accept"]
