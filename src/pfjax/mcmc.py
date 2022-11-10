"""
MCMC algorithms for state space models.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import lax


class AdaptiveMWG:
    def __init__(self, adapt_max=.01, adapt_rate=.5):
        r"""
        Adaptive Metropolis-within-Gibbs.

        Notes:

        - Design heavily inspired by [BlackJAX](https://blackjax-devs.github.io/blackjax/index.html).  Perhaps to be fully integrated with it some day.

        Args:
            adapt_max: Scalar or vector of maximum adaptation amounts.
            adapt_rate: Scalar or vector of adaptation rates.
        """
        # self._n_params = n_params
        self._adapt_max = adapt_max
        self._adapt_rate = adapt_rate
        self._targ_acc = .44

    def adapt_sd(self, rw_sd, n_iter, accept_rate):
        r"""
        Update random walk standard deviations.

        Args:
            rw_sd: Vector of standard deviations for the random walk proposal on each component of `position`.
            n_iter: Number of MWG iterations (cycles) so far.
            accept_rate: Vector of `n_params` acceptance rates.

        Returns:
            Vector of updated random walk standard deviations.
        """
        delta = jnp.power(n_iter, -self._adapt_rate)
        delta = jnp.minimum(delta, self._adapt_max)
        low_acc = jnp.sign(self._targ_acc - accept_rate)
        return jnp.exp(jnp.log(rw_sd) - delta * low_acc)

    def init(self, position, rw_sd):
        r"""
        Initialize the state of the sampler.

        Args:
            position: The initial position of the sampler.
            rw_sd: A vector of the same length as `position` of initial standard deviations for the componentwise random walk proposal.

        Returns:
            A dictionary with elements

                - `position`: The initial position of the sampler.
                - `rw_sd`: The vector of  standard deviations for the componentwise random walk proposal.
                - `n_iter`: The number of MWG steps taken so far, which is zero.
                - `n_accept`: The number of draws of each component accepted so far, which is ``jnp.zeros_like(position)``.
        """
        return {
            "position": position,
            "rw_sd": rw_sd,
            "n_iter": 0.,
            "n_accept": jnp.zeros_like(position)
        }

    def update(self, key, state, logprob_fn, order=None):
        r"""
        Update parameters via adaptive Metropolis-within-Gibbs.

        Args:
            key: PRNG key.
            state: The current state of the sampler.  A dictionary with elements

                - `position`: The current position of the sampler.
                - `rw_sd`: Vector of standard deviations for the componentwise random walk proposal.
                - `n_iter`: The number of MWG steps taken so far.
                - `n_accept`: The number of draws of each component accepted so far.

            logprob_fn: Function which takes a JAX array input `position` and returns a scalar corresponding to the log of the probability density at that input.
            order: Optional vector of integers between 0 and ``position.size`` indicating the order in which to update the components of `position`.  Can use this to keep certain components fixed, randomize update order, etc.

        Returns:
            Tuple with elements

            - **state** - The updated state dictionary.
            - **accept** - Boolean vector of length ``position.size`` indicating whether or not each proposal was accepted.
        """
        n_pos = state["position"].size  # number of components to update
        rw_sd = state["rw_sd"]
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
            "position": state["position"],
            "lp": logprob_fn(state["position"]),
            "key": key
        }
        # scan itself
        last, full = jax.lax.scan(fun, init, order)
        position = last["position"]
        accept = full["accept"]
        # new adapt_pars
        n_iter = state["n_iter"] + 1.
        n_accept = state["n_accept"] + accept
        rw_sd = self.adapt_sd(
            rw_sd=rw_sd,
            n_iter=n_iter,
            accept_rate=(1. * n_accept) / n_iter
        )
        state = {
            "position": position,
            "rw_sd": rw_sd,
            "n_iter": n_iter,
            "n_accept": n_accept
        }
        return state, accept
