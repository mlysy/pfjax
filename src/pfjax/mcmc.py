"""
MCMC algorithms for state space models.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import lax


class AdaptiveMWG:
    r"""
    Adaptive Metropolis-within-Gibbs.

    **Notes:**

    Design heavily inspired by [BlackJAX](https://blackjax-devs.github.io/blackjax/index.html).  Perhaps to be fully integrated with it some day.

    Args:
        adapt_max: Scalar or vector of maximum adaptation amounts.
        adapt_rate: Scalar or vector of adaptation rates.
    """

    def __init__(self, adapt_max=.01, adapt_rate=.5):
        # self._n_params = n_params
        self._adapt_max = adapt_max
        self._adapt_rate = adapt_rate
        self._targ_acc = .44

    def adapt(self, pars, accept):
        r"""
        Update random walk standard deviations.

        Args:
            pars: Adaptation parameters.  A dictionary with elements

                - `rw_sd`: Vector of standard deviations for the random walk proposal on each component of `position`.
                - `n_iter`: Number of MWG iterations (cycles) so far.
                - `n_accept`: The number of accepted draws per component so far.

            accept: Boolean vector indicating whether or not the latest proposal was accepted.

        Returns:
            Dictionary with elements

            - **rw_sd** - Vector of updated random walk standard deviations.
            - **n_iter** - Updated number of iterations, i.e., ``n_iter += 1``.
            - **n_accept** - Updated number of accepted draws, i.e., ``n_accept += accept``.
        """
        rw_sd = pars["rw_sd"]
        n_iter = pars["n_iter"] + 1.
        n_accept = pars["n_accept"] + accept
        accept_rate = (1. * n_accept) / n_iter
        delta = jnp.power(n_iter, -self._adapt_rate)
        delta = jnp.minimum(delta, self._adapt_max)
        low_acc = jnp.sign(self._targ_acc - accept_rate)
        return {
            "rw_sd": jnp.exp(jnp.log(rw_sd) - delta * low_acc),
            "n_iter": n_iter,
            "n_accept": n_accept
        }

    def init(self, rw_sd):
        r"""
        Initialize the adaptation parameters.

        Args:
            rw_sd: A vector of initial standard deviations for the componentwise random walk proposal.

        Returns:
            A dictionary with elements

                - `rw_sd`: The vector of  standard deviations for the componentwise random walk proposal.
                - `n_iter`: The number of MWG steps taken so far, which is zero.
                - `n_accept`: The number of draws of each component accepted so far, which is ``jnp.zeros_like(rw_sd)``.
        """
        return {
            "rw_sd": rw_sd,
            "n_iter": 0.,
            "n_accept": jnp.zeros_like(rw_sd)
        }

    def step(self, key, position, logprob_fn, rw_sd, order=None):
        r"""
        Update parameters via adaptive Metropolis-within-Gibbs.

        Args:
            key: PRNG key.
            position: The current position of the sampler.
            logprob_fn: Function which takes a JAX array input `position` and returns a scalar corresponding to the log of the probability density at that input.
            rw_sd: Vector of standard deviations for the componentwise random walk proposal.
            order: Optional vector of integers between 0 and ``position.size`` indicating the order in which to update the components of `position`.  Can use this to keep certain components fixed, randomize update order, etc.

        Returns:
            Tuple with elements

            - **position** - The updated position.
            - **accept** - Boolean vector of length ``position.size`` indicating whether or not each proposal was accepted.
        """
        n_pos = position.size  # number of components to update
        if order is None:
            order = jnp.arange(n_pos)

        # lax.scan setup
        def fun(carry, i):
            lp_curr = carry["lp"]
            position_curr = carry["position"]
            key = carry["key"]
            # 2 subkeys for each position: rw_jump and mh_accept
            key, *subkeys = random.split(key, num=3)
            # proposal
            position_prop = position_curr.at[i].set(
                position_curr[i] + rw_sd[i] * random.normal(key=subkeys[0])
            )
            # acceptance rate
            lp_prop = logprob_fn(position_prop)
            lrate = lp_prop - lp_curr
            # update parameter draw
            acc = random.bernoulli(key=subkeys[1],
                                   p=jnp.minimum(1.0, jnp.exp(lrate)))
            res = {
                "position": position_curr.at[i].set(
                    position_prop[i] * acc + position_curr[i] * (1-acc)
                ),
                "lp": lp_prop * acc + lp_curr * (1-acc),
                # "accept": acc,
                "key": key
            }
            stack = {"accept": acc}
            return res, stack
        # scan initial value
        init = {
            "position": position,
            "lp": logprob_fn(position),
            "key": key
        }
        # scan itself
        last, full = jax.lax.scan(fun, init, order)
        position = last["position"]
        accept = full["accept"]
        return position, accept
