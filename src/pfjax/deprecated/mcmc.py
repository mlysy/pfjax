"""
MCMC algorithms for state space models.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import lax
from jax.experimental.maps import xmap
from .loglik_full import *


# --- a few priors -------------------------------------------------------------


class FlatPrior:
    """
    Flat prior class.
    """

    def lpdf(self, theta):
        """
        Calculate the log-pdf.

        Args:
            theta: Parameter value.

        Returns:
            The log-pdf of the prior.
        """
        return jnp.array(0.)


class NormalDiagPrior(object):
    def __init__(self, loc, scale):
        """
        Normal prior class with diagonal variance matrix.

        Args:
            loc: Vector of means.
            scale: Vector of standard deviations.
        """
        self._loc = loc
        self._scale = scale

    def lpdf(self, theta):
        """
        Calculate the log-pdf.

        Args:
            theta: Parameter value.

        Returns:
            The log-pdf of the prior.
        """
        return jnp.sum(jsp.stats.norm.logpdf(theta,
                                             loc=self._loc, scale=self._scale))


# --- parameter updates --------------------------------------------------------


def param_mwg_update(model, prior, key, theta, x_state, y_meas, rw_sd, theta_order):
    """
    Parameter update by Metropolis-within-Gibbs random walk.

    **Notes:**

    - Assumes the parameters are real valued.  Next step might be to provide a parameter validator to the model.
    - Potentially wastes an initial evaluation of `loglik_full(theta)`.  Could be passed in from a previous calculation but a bit cumbersome.

    Args:
        model: Object specifying the state-space model.
        prior: Object specifying the parameter prior.
        key: PRNG key.
        theta: Current parameter vector.
        x_state: The sequence of `n_obs` state variables `x_state = (x_0, ..., x_T)`, where `T = n_obs-1`.
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`.
        rw_sd: Vector of length `n_param = theta.size` standard deviations for the componentwise random walk proposal.
        theta_order: Vector of integers between 0 and `n_param-1` indicating the order in which to update the components of `theta`.  Can use this to keep certain components fixed.

    Returns:
        theta_out: Updated parameter vector.
        accept: Boolean vector of size `theta_order.size` indicating whether or not the proposal was accepted. 
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
        lp_prop = loglik_full(model, y_meas, x_state,
                              theta_prop) + prior.lpdf(theta_prop)
        lrate = lp_prop - lp_curr
        # update parameter draw
        acc = random.bernoulli(key=subkeys[1],
                               p=jnp.minimum(1.0, jnp.exp(lrate)))
        res = {
            "theta_curr": theta_curr.at[i].set(
                theta_prop[i] * acc + theta_curr[i] * (1-acc)
            ),
            "lp_curr": lp_prop * acc + lp_curr * (1-acc),
            "accept": acc,
            "key": key
        }
        return res, res
    # scan initial value
    init = {
        "theta_curr": theta,
        "lp_curr": loglik_full(model, y_meas, x_state,
                               theta) + prior.lpdf(theta),
        "accept": jnp.array(True),
        "key": key
    }
    # scan itself
    last, full = lax.scan(fun, init, theta_order)
    return last["theta_curr"], full["accept"]


def mwg_adapt(rw_sd, accept_rate, n_iter,
              adapt_max=.01, adapt_rate=.5):
    r"""
    Adapt random walk jump sizes of MWG proposals.

    Given a vector of random walk jump sizes, increase or decrease each of them depending on whether the cumulative acceptance rate is above or below 0.44.  The amount of change on log-scale is 

    ```
    delta = min(adapt_max, 1/n_iter^adapt_rate)
    ```

    Args:
        rw_sd: Vector of `n_params` standard deviations (jump sizes) for the componentwise random walk proposal.
        accept_rate: Vector of `n_params` cumulative acceptance rates (i.e., between 0 and 1).
        n_iter: Number of MCMC iterations so far.
        adapt_max: Scalar or vector of `n_params` maximum adaptation amounts.
        adapt_rate: Scalar or vector of `n_params` adaptation rates.

    Returns:
        Vector of `n_params` adapted standard deviations.
    """

    targ_acc = 0.44  # target acceptance rate
    delta = jnp.power(n_iter, -adapt_rate)
    delta = jnp.minimum(delta, adapt_max)
    low_acc = jnp.sign(targ_acc - accept_rate)
    return jnp.exp(jnp.log(rw_sd) - delta * low_acc)
