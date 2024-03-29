"""
Particle filters which approximate the score and fisher information.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from jax import random
from jax import lax
# from jax.experimental.host_callback import id_print
from .utils import *
from .particle_resamplers import resample_multinomial


def particle_filter(model, key, y_meas, theta, n_particles,
                    resampler=resample_multinomial,
                    score=False, fisher=False, history=False):
    r"""
    Basic particle filter.

    **Notes:**

    - Can optionally use an auxiliary particle filter if `model.pf_aux()` is provided.

    - The score and fisher information are estimated using the method described by Poyiadjis et al 2011, Algorithm 1. 

    - Should have the option of adding new measurements as we go along.  So for example, could have an argument `init=None`, which if not `None` is the carry from `lax.scan()`.  Should then also return the carry as an output...

    Args:
        model: Object specifying the state-space model having the following methods:

            - `pf_init : (key, y_init, theta) -> (x_particles, logw)`: For sampling and calculating log-weights for the initial latent variable.

            - `pf_step : (key, x_prev, y_curr, theta) -> (x_particles, logw)`: For sampling and calculating log-weights for each subsequent latent variable.

            - `pf_aux : (x_prev, y_curr, theta) -> logw`: Optional method providing look-forward log-weights of the auxillary particle filter.

            - `state_lpdf : (x_curr, x_prev, theta) -> lpdf`: Optional method specifying the log-density of the state model.  Only required if `score or fisher == True`.

            - `meas_lpdf : (y_curr, x_curr, theta) -> lpdf`: Optional method specifying the log-density of the measurement model.  Only required if `score or fisher == True`.

        key: PRNG key.
        y_meas: JAX array with leading dimension `n_obs` containing the measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        n_particles: Number of particles.
        resampler: Function used at step `t` to obtain sample of particles from `p(x_{t} | y_{0:t}, theta)` out of a sample of particles from `p(x_{t-1} | y_{0:t-1}, theta)`.   The argument signature is `resampler(x_particles, logw, key)`, and the return value is a dictionary with mandatory element `x_particles`  and optional elements that get carried to the next step `t+1` via `lax.scan()`.
        score: Whether or not to return an estimate of the score function at `theta`.  Only works if `resampler` has an output element named `ancestors`.
        fisher: Whether or not to return an estimate of the Fisher information at `theta`.  Only works if `resampler` has an output element named `ancestors`.  If `True` returns score as well.
        history: Whether to output the history of the filter or only the last step.

    Returns:
        Dictionary:

        - **x_particles** - JAX array containing the state variable particles at the last time point (leading dimension `n_particles`) or at all time points (leading dimensions `(n_obs, n_particles)` if `history=True`.
        - **logw** - JAX array containing unnormalized log weights at the last time point (dimensions `n_particles`) or at all time points (dimensions (n_obs, n_particles)`) if `history=True`.
        - **loglik** - The particle filter loglikelihood evaluated at `theta`.
        - **score** - Optional 1D JAX array of size `n_theta = length(theta)` containing the estimated score at `theta`.
        - **fisher** - Optional JAX array of shape `(n_theta, n_theta)` containing the estimated observed fisher information at `theta`.
        - **resample_out** - If `history=True`, a dictionary of additional outputs from `resampler` function.  The leading dimension of each element of the dictionary has leading dimension `n_obs-1`, since these additional outputs do not apply to the first time point `t=0`.
    """
    n_obs = y_meas.shape[0]
    has_acc = score or fisher

    # accumulator for derivatives
    def accumulator(x_prev, x_curr, y_curr, acc_prev):
        """
        Accumulator for derivative calculations.

        Args:
            acc_prev: Dictionary with elements alpha and optionally beta.

        Returns:
            Dictionary with elements:
            - logw_targ: logw_bar + state_lpdf.
            - logw_prop: logw_aux + prop_lpdf.
            - alpha: optional current value of alpha_bar.
            - beta: optional current value of beta_bar.
        """
        grad_meas = jax.grad(model.meas_lpdf, argnums=2)
        grad_state = jax.grad(model.state_lpdf, argnums=2)
        alpha = grad_meas(y_curr, x_curr, theta) + \
            grad_state(x_curr, x_prev, theta) + \
            acc_prev["alpha"]
        acc_curr = {"alpha": alpha}
        if fisher:
            hess_meas = jax.jacfwd(jax.jacrev(model.meas_lpdf, argnums=2),
                                   argnums=2)
            hess_state = jax.jacfwd(jax.jacrev(model.state_lpdf, argnums=2),
                                    argnums=2)
            beta = hess_meas(y_curr, x_curr, theta) + \
                hess_state(x_curr, x_prev, theta) + \
                acc_prev["beta"]
            acc_curr["beta"] = beta
        return acc_curr

    # internal functions for vmap
    def pf_step(key, x_prev, y_curr):
        return model.pf_step(key=key, x_prev=x_prev, y_curr=y_curr, theta=theta)

    def pf_init(key):
        return model.pf_init(key=key, y_init=y_meas[0], theta=theta)

    def pf_aux(logw_prev, x_prev, y_curr):
        """
        Add the log-density for auxillary sampling `logw_aux` to the log-weights from the previous step `logw_prev`.

        The auxillary log-density is given by model.pf_aux(). If this method is missing, `logw_aux` is set to 0.
        """
        if callable(getattr(model, "pf_aux", None)):
            logw_aux = model.pf_aux(
                x_prev=x_prev,
                y_curr=y_curr,
                theta=theta
            )
            return logw_aux + logw_prev
        else:
            return logw_prev

    def pf_acc(acc_prev, x_prev, x_curr, y_curr):
        acc_curr = accumulator(
            x_prev=x_prev, x_curr=x_curr, y_curr=y_curr, theta=theta
        )
        return tree_add(tree1=acc_prev, tree2=acc_curr)

    # lax.scan stepping function
    def filter_step(carry, t):
        # 1. sample particles from previous time point
        key, subkey = random.split(carry["key"])
        # augment previous weights with auxiliary weights
        logw_aux = jax.vmap(
            pf_aux,
            in_axes=(0, 0, None)
        )(carry["logw"], carry["x_particles"], y_meas[t])
        # resampled particles
        resample_out = resampler(
            key=subkey,
            x_particles_prev=carry["x_particles"],
            logw=logw_aux
        )
        # 2. sample particles for current time point
        key, *subkeys = random.split(key, num=n_particles+1)
        x_particles, logw = jax.vmap(
            pf_step,
            in_axes=(0, 0, None)
        )(jnp.array(subkeys), resample_out["x_particles"], y_meas[t])
        if has_acc:
            # 3. accumulate values for score and/or fisher information
            acc_prev = {"alpha": carry["alpha"]}
            if fisher:
                acc_prev["beta"] = carry["beta"]
            # resample acc_prev
            acc_prev = tree_shuffle(
                tree=acc_prev,
                index=resample_out["ancestors"]
            )
            # add new values to get acc_curr
            acc_curr = jax.vmap(
                accumulator,
                in_axes=(0, 0, None, 0)
            )(resample_out["x_particles"], x_particles, y_meas[t], acc_prev)
        # output
        res_carry = {
            "x_particles": x_particles,
            "logw": logw,
            "key": key,
            "loglik": carry["loglik"] + jsp.special.logsumexp(logw)
        }
        if has_acc:
            res_carry.update(acc_curr)
        if history:
            res_stack = {k: res_carry[k]
                         for k in ["x_particles", "logw"]}
            if set(["x_particles"]) < resample_out.keys():
                res_stack["resample_out"] = rm_keys(
                    x=resample_out,
                    keys="x_particles"
                )
        else:
            res_stack = None
        return res_carry, res_stack

    # lax.scan initial value
    key, *subkeys = random.split(key, num=n_particles+1)
    # initial particles and weights
    x_particles, logw = jax.vmap(
        pf_init
    )(jnp.array(subkeys))
    filter_init = {
        "x_particles": x_particles,
        "logw": logw,
        "loglik": jsp.special.logsumexp(logw),
        "key": key,
    }
    if has_acc:
        # dummy initialization for derivatives
        filter_init["alpha"] = jnp.zeros((n_particles, theta.size))
        if fisher:
            filter_init["beta"] = jnp.zeros(
                (n_particles, theta.size, theta.size)
            )
    # # dummy initialization for resampler
    # init_res = resampler(key, x_particles, logw)
    # init_res = tree_zeros(rm_keys(init_res, ["x_particles"]))
    # if set("x_particles") < init_res.keys():
    #     filter_init["resample_out"]
    # if has_acc:
    #     # dummy initialization for accumulator
    #     init_acc = jax.vmap(
    #         accumulator,
    #         in_axes=(0, 0, None, None)
    #     )(x_particles, x_particles, y_meas[0], theta)
    #     init_acc = tree_zeros(init_acc)
    #     filter_init.update(init_acc)

    # lax.scan itself
    last, full = lax.scan(filter_step, filter_init, jnp.arange(1, n_obs))

    # format output
    if history:
        # append initial values of x_particles and logw
        full["x_particles"] = jnp.concatenate([
            filter_init["x_particles"][None], full["x_particles"]
        ])
        full["logw"] = jnp.concatenate([
            filter_init["logw"][None], full["logw"]
        ])
        # if has_acc:
        #     # append derivative calculations
        #     full["score"] = last["score"]
    else:
        full = last

    # calculate loglikelihood
    full["loglik"] = last["loglik"] - n_obs * jnp.log(n_particles)
    if has_acc:
        if not fisher:
            # calculate score only
            full["score"] = tree_mean(last["alpha"], last["logw"])
            # alpha = jax.vmap(
            #     jnp.multiply
            # )(prob, full["score"])
            # full["score"] = jnp.sum(alpha, axis=0)
        else:
            # calculate score and fisher information
            prob = logw_to_prob(last["logw"])
            alpha = last["alpha"]
            beta = last["beta"]
            alpha, gamma = jax.vmap(
                lambda p, a, b: (p * a, p * (jnp.outer(a, a) + b))
            )(prob, alpha, beta)
            alpha = jnp.sum(alpha, axis=0)
            hess = jnp.sum(gamma, axis=0) - jnp.outer(alpha, alpha)
            full["score"] = alpha
            full["fisher"] = -hess
    return full


def particle_filter_rb(model, key, y_meas, theta, n_particles,
                       resampler=resample_multinomial,
                       score=False, fisher=False, history=False):
    r"""
    Rao-Blackwellized particle filter.

    Notes:

        - Algorithm 2 of Poyiadjis et al 2011.

        - Can optionally use an auxiliary particle filter if `model.pf_aux()` is provided.

    Args:
        model: Object specifying the state-space model having the following methods:

            - `pf_init : (key, y_init, theta) -> (x_particles, logw)`: For sampling and calculating log-weights for the initial latent variable.
            - `step_sample : (key, x_prev, y_curr, theta) -> x_curr`: Sampling from the proposal distribution for each subsequent latent variable.
            - `step_lpdf : (x_curr, x_prev, y_curr, theta) -> logw`: Calculate log-weights for each subsequent latent variable.
            - `state_lpdf : (x_curr, x_prev, theta) -> lpdf`: Calculates the log-density of the state model.
            - `meas_lpdf : (y_curr, x_curr, theta) -> lpdf`: Calculates the log-density of the measurement model.
            - `pf_aux : (x_prev, y_curr, theta) -> logw`: Optional method providing look-forward log-weights of the auxillary particle filter.

        key: PRNG key.
        y_meas: JAX array with leading dimension `n_obs` containing the measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        n_particles: Number of particles.
        resampler: Function used at step `t` to obtain sample of particles from `p(x_{t} | y_{0:t}, theta)` out of a sample of particles from `p(x_{t-1} | y_{0:t-1}, theta)`.   The argument signature is `resampler(x_particles, logw, key)`, and the return value is a dictionary with mandatory element `x_particles`  and optional elements that get carried to the next step `t+1` via `lax.scan()`.
        score: Whether or not to return an estimate of the score function at `theta`.
        fisher: Whether or not to return an estimate of the Fisher information at `theta`.  If `True` returns score as well.
        history: Whether to output the history of the filter or only the last step.

    Returns:
        Dictionary:

        - **x_particles** - JAX array containing the state variable particles at the last time point (leading dimension `n_particles`) or at all time points (leading dimensions `(n_obs, n_particles)` if `history=True`.
        - **logw_bar** - JAX array containing unnormalized log weights at the last time point (dimensions `n_particles`) or at all time points (dimensions (n_obs, n_particles)`) if `history=True`.
        - **loglik** - The particle filter loglikelihood evaluated at `theta`.
        - **score** - Optional 1D JAX array of size `n_theta = length(theta)` containing the estimated score at `theta`.
        - **fisher** Optional JAX array of shape `(n_theta, n_theta)` containing the estimated observed fisher information at `theta`.
        - **resample_out** If `history=True`, a dictionary of additional outputs from `resampler` function.  The leading dimension of each element of the dictionary has leading dimension `n_obs-1`, since these additional outputs do not apply to the first time point `t=0`.
    """
    n_obs = y_meas.shape[0]
    has_acc = score or fisher

    # internal functions for vmap
    def accumulator(x_prev, x_curr, y_curr, acc_prev, logw_aux):
        """
        Accumulator for weights and possibly derivative calculations.

        Args:
            acc_prev: Dictionary with elements logw_bar, and optionally alpha and beta.

        Returns:
            Dictionary with elements:
            - logw_targ: logw_bar + state_lpdf.
            - logw_prop: logw_aux + prop_lpdf.
            - alpha: optional current value of alpha_bar.
            - beta: optional current value of beta_bar.
        """
        logw_bar = acc_prev["logw_bar"]
        logw_targ = model.state_lpdf(x_curr=x_curr, x_prev=x_prev,
                                     theta=theta) + \
            logw_bar
        logw_prop = model.step_lpdf(x_curr=x_curr, x_prev=x_prev,
                                    y_curr=y_curr, theta=theta) + \
            logw_aux
        acc_full = {"logw_targ": logw_targ, "logw_prop": logw_prop}
        if has_acc:
            grad_meas = jax.grad(model.meas_lpdf, argnums=2)
            grad_state = jax.grad(model.state_lpdf, argnums=2)
            alpha = grad_meas(y_curr, x_curr, theta) + \
                grad_state(x_curr, x_prev, theta) + \
                acc_prev["alpha"]
            acc_full["alpha"] = alpha
            if fisher:
                hess_meas = jax.jacfwd(jax.jacrev(model.meas_lpdf,
                                                  argnums=2),
                                       argnums=2)
                hess_state = jax.jacfwd(jax.jacrev(model.state_lpdf,
                                                   argnums=2),
                                        argnums=2)
                beta = jnp.outer(alpha, alpha) + \
                    hess_meas(y_curr, x_curr, theta) + \
                    hess_state(x_curr, x_prev, theta) + \
                    acc_prev["beta"]
                acc_full["beta"] = beta
        return acc_full

    def pf_prop(x_curr, x_prev, y_curr):
        """
        Calculate log-density of the proposal distribution `x_curr ~ q(x_t | x_t-1, y_t, theta)`.
        """
        return model.step_lpdf(x_curr=x_curr, x_prev=x_prev, y_curr=y_curr,
                               theta=theta)

    def pf_step(key, x_prev, y_curr):
        """
        Sample from the proposal distribution `x_curr ~ q(x_t | x_t-1, y_t, theta)`.
        """
        return model.step_sample(key=key, x_prev=x_prev, y_curr=y_curr,
                                 theta=theta)
        # if callable(getattr(model, "pf_prop", None)):
        #     x_curr = model.pf_prop(key=key, x_prev=x_prev, y_curr=y_curr,
        #                            theta=theta)
        # else:
        #     x_curr, _ = model.pf_step(key=key, x_prev=x_prev, y_curr=y_curr,
        #                               theta=theta)
        # return x_curr

    def pf_init(key):
        return model.pf_init(key=key, y_init=y_meas[0], theta=theta)

    def pf_aux(logw_prev, x_prev, y_curr):
        """
        Add the log-density for auxillary sampling `logw_aux` to the log-weights from the previous step `logw_prev`.

        The auxillary log-density is given by model.pf_aux(). If this method is missing, `logw_aux` is set to 0.
        """
        if callable(getattr(model, "pf_aux", None)):
            logw_aux = model.pf_aux(
                x_prev=x_prev,
                y_curr=y_curr,
                theta=theta
            )
            return logw_aux + logw_prev
        else:
            return logw_prev

    def pf_bar(x_prev, x_curr, y_curr, acc_prev, logw_aux):
        """
        Update calculations relating to logw_bar.

        This is the vmap version, which does the loop over x_prev here and the loop over x_curr in filter_step.

        Returns:
            Dictionary with elements:
            - logw_bar: rao-blackwellized weights.
            - alpha: optional current value of alpha.
            - beta: optional current value of beta.
        """
        # Calculate the accumulated values looping over x_prev
        acc_full = jax.vmap(
            accumulator,
            in_axes=(0, None, None, 0, 0)
        )(x_prev, x_curr, y_curr, acc_prev, logw_aux)
        # Calculate logw_tilde and logw_bar
        logw_targ, logw_prop = acc_full["logw_targ"], acc_full["logw_prop"]
        logw_tilde = jsp.special.logsumexp(logw_targ) + \
            model.meas_lpdf(
            y_curr=y_curr,
            x_curr=x_curr,
            theta=theta
        )
        logw_bar = logw_tilde - jsp.special.logsumexp(logw_prop)
        acc_curr = {"logw_bar": logw_bar}
        if has_acc:
            # weight the derivatives
            grad_curr = tree_mean(
                tree=rm_keys(acc_full, ["logw_targ", "logw_prop"]),
                logw=logw_targ
            )
            if fisher:
                grad_curr["beta"] = grad_curr["beta"] - \
                    jnp.outer(grad_curr["alpha"], grad_curr["alpha"])
            acc_curr.update(grad_curr)
        return acc_curr

    # lax.scan stepping function
    def filter_step(carry, t):
        # 1. sample particles from previous time point
        key, subkey = random.split(carry["key"])
        # augment previous weights with auxiliary weights
        logw_aux = jax.vmap(
            pf_aux,
            in_axes=(0, 0, None)
        )(carry["logw_bar"], carry["x_particles"], y_meas[t])
        # resampled particles
        resample_out = resampler(
            key=subkey,
            x_particles_prev=carry["x_particles"],
            logw=logw_aux
        )
        # 2. sample particles for current timepoint
        key, *subkeys = random.split(key, num=n_particles+1)
        x_particles = jax.vmap(
            pf_step,
            in_axes=(0, 0, None)
        )(jnp.array(subkeys), resample_out["x_particles"], y_meas[t])
        # 3. compute all double summations
        acc_prev = {"logw_bar": carry["logw_bar"]}
        if has_acc:
            acc_prev["alpha"] = carry["alpha"]
            if fisher:
                acc_prev["beta"] = carry["beta"]
        acc_curr = jax.vmap(
            pf_bar,
            in_axes=(None, 0, None, None, None)
        )(carry["x_particles"], x_particles, y_meas[t],
          acc_prev, logw_aux)
        # output
        res_carry = {
            "x_particles": x_particles,
            "loglik": carry["loglik"] +
            jsp.special.logsumexp(acc_curr["logw_bar"]),
            "key": key
            # "resample_out": rm_keys(resample_out, "x_particles")
        }
        res_carry.update(acc_curr)
        if history:
            res_stack = {k: res_carry[k]
                         for k in ["x_particles", "logw_bar"]}
            if set(["x_particles"]) < resample_out.keys():
                res_stack["resample_out"] = rm_keys(
                    x=resample_out,
                    keys="x_particles"
                )
        else:
            res_stack = None
        return res_carry, res_stack

    # lax.scan initial value
    key, *subkeys = random.split(key, num=n_particles+1)
    # initial particles and weights
    x_particles, logw = jax.vmap(
        pf_init
    )(jnp.array(subkeys))
    # # dummy initialization for resampler
    # init_res = resampler(key, x_particles, logw)
    # init_res = tree_zeros(rm_keys(init_res, ["x_particles"]))
    filter_init = {
        "x_particles": x_particles,
        "loglik": jsp.special.logsumexp(logw),
        "logw_bar": logw,
        "key": key
        # "resample_out": init_res
    }
    if has_acc:
        # dummy initialization for derivatives
        filter_init["alpha"] = jnp.zeros((n_particles, theta.size))
        if fisher:
            filter_init["beta"] = jnp.zeros(
                (n_particles, theta.size, theta.size)
            )

    # lax.scan itself
    last, full = lax.scan(filter_step, filter_init, jnp.arange(1, n_obs))

    # format output
    if history:
        # append initial values of x_particles and logw
        full["x_particles"] = jnp.concatenate([
            filter_init["x_particles"][None], full["x_particles"]
        ])
        full["logw_bar"] = jnp.concatenate([
            filter_init["logw_bar"][None], full["logw_bar"]
        ])
    else:
        full = last
    # calculate loglikelihood
    full["loglik"] = last["loglik"] - n_obs * jnp.log(n_particles)
    if has_acc:
        # logw_bar = last["logw_bar"]
        if not fisher:
            # calculate score only
            full["score"] = tree_mean(last["alpha"], last["logw_bar"])
        else:
            # calculate score and fisher information
            prob = logw_to_prob(last["logw_bar"])
            alpha = last["alpha"]
            beta = last["beta"]
            alpha, gamma = jax.vmap(
                lambda p, a, b: (p * a, p * (jnp.outer(a, a) + b))
            )(prob, alpha, beta)
            alpha = jnp.sum(alpha, axis=0)
            hess = jnp.sum(gamma, axis=0) - jnp.outer(alpha, alpha)
            full["score"] = alpha
            full["fisher"] = -hess

    return full
