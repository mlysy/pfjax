"""
Particle filters which approximate the score and fisher information.

The API requires the user to define a model class with the following methods:

- `pf_init: (key, y_init, theta) => (x_particles, logw)`: Sampling and log-weights for the initial latent variable.

- `pf_step: (key, x_prev, y_curr, theta) => (x_particles, logw)`: Sampling and log-weights for each subsequent latent variable.

- `pf_aux: (x_prev, y_curr, theta) => logw`: The look-forward log-weights of the auxillary particle filter.

- `state_lpdf: (x_curr, x_prev, theta) => lpdf`: The log-density of the state model.

- `meas_lpdf: (y_curr, x_curr, theta) => lpdf`: The log-density of the measurement model.

For now the resampling function is just the multinomial, but we'll keep the `resampler` argument to eventually pass in other resamplers with `ancestors`.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from jax import random
from jax import lax
# from jax.experimental.host_callback import id_print
from ..utils import *
from ..particle_resamplers import resample_multinomial


def particle_filter(model, key, y_meas, theta, n_particles,
                    resampler=resample_multinomial,
                    score=True, fisher=False, history=False):
    r"""
    Basic particle filter.

    **FIXME:**

    - Resample outputs should go to `resample_out` instead of directly to output.  This avoids resampler overwriting dict elements set by particle_filter.  This is true even if using `resampler_multinomial`.

    Args:
        model: Object specifying the state-space model.
        key: PRNG key.
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        n_particles: Number of particles.
        resampler: Resampling function.  Argument signature is `resampler(key, x_particles_prev, logw_prev, logw_aux) => (x_particles_curr, ancestors)`.
        score: Whether or not to return an estimate of the score function at `theta`.
        fisher: Whether or not to return an estimate of the Fisher information at `theta`.  If `True` returns score as well.
        history: Whether to output the history of the filter or only the last step.

    Returns:
        A dictionary with elements:
            - `loglik`: The particle filter loglikelihood evaluated at `theta`.
            - `score`: A vector of size `n_theta = length(theta)` containing the estimated score at `theta`.
            - `fisher`: An array of size `(n_theta, n_theta)` containing the estimated observed fisher information at `theta`.
            - `x_particles`: A jax array containing the state variable particles at the last time point (leading dimension `n_particles`) or at all time points (leading dimensions `(n_obs, n_particles)` if `history=True`.
            - `logw`: A jax array containing unnormalized log weights at the last time point (dimensions `n_particles`) or at all time points (dimensions (n_obs, n_particles)`) if `history=True`.
            - `ancestors`: If `history=True`, an array of shape `(n_obs-1, n_particles)` containing the ancestor of each particle at times `t=1,...,T`.
    """
    n_obs = y_meas.shape[0]
    has_acc = score or fisher

    # accumulator for derivatives
    def accumulator(x_prev, x_curr, y_curr, theta):
        grad_meas = jax.grad(model.meas_lpdf, argnums=2)
        grad_state = jax.grad(model.state_lpdf, argnums=2)
        alpha = grad_meas(y_curr, x_curr, theta) + \
            grad_state(x_curr, x_prev, theta)
        if fisher:
            hess_meas = jax.jacfwd(jax.jacrev(model.meas_lpdf, argnums=2),
                                   argnums=2)
            hess_state = jax.jacfwd(jax.jacrev(model.state_lpdf, argnums=2),
                                    argnums=2)
            beta = hess_meas(y_curr, x_curr, theta) + \
                hess_state(x_curr, x_prev, theta)
            return (alpha, beta)
        else:
            return alpha

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
        # sample particles from previous time point
        key, subkey = random.split(carry["key"])
        logw_aux = jax.vmap(
            pf_aux,
            in_axes=(0, 0, None)
        )(carry["logw"], carry["x_particles"], y_meas[t])
        new_particles = resampler(
            key=subkey,
            x_particles_prev=carry["x_particles"],
            logw=logw_aux
        )
        # update particles to current time point (and get weights)
        key, *subkeys = random.split(key, num=n_particles+1)
        x_particles, logw = jax.vmap(
            pf_step,
            in_axes=(0, 0, None)
        )(jnp.array(subkeys), new_particles["x_particles"], y_meas[t])
        if has_acc:
            # accumulate expectation
            # note: we're calling the accumulated value "score"
            # so we don't need to rename the dictionary item later.
            acc_prev = carry["score"]
            # resample acc_prev
            acc_prev = tree_shuffle(
                tree=acc_prev,
                index=new_particles["ancestors"]
            )
            acc_curr = jax.vmap(
                pf_acc,
                in_axes=(0, 0, 0, None)
            )(acc_prev, new_particles["x_particles"], x_particles, y_meas[t])
        # output
        res_carry = {
            "x_particles": x_particles,
            "logw": logw,
            "key": key,
            "loglik": carry["loglik"] + jsp.special.logsumexp(logw),
            "ancestors": new_particles["ancestors"]
        }
        if has_acc:
            res_carry["score"] = acc_curr
        res_stack = rm_keys(
            res_carry, ["key", "loglik", "score"]
        ) if history else None
        return res_carry, res_stack

    # lax.scan initial value
    key, *subkeys = random.split(key, num=n_particles+1)
    # initial particles and weights
    x_particles, logw = jax.vmap(
        pf_init
    )(jnp.array(subkeys))
    # dummy initialization for ancestors
    init_ancestors = jnp.array([0] * n_particles)
    if has_acc:
        # dummy initialization for accumulate
        init_acc = jax.vmap(
            accumulator,
            in_axes=(0, 0, None, None)
        )(x_particles, x_particles, y_meas[0], theta)
        init_acc = tree_zeros(init_acc)
    filter_init = {
        "x_particles": x_particles,
        "logw": logw,
        "loglik": jsp.special.logsumexp(logw),
        "key": key,
        "ancestors": init_ancestors
    }
    if has_acc:
        filter_init["score"] = init_acc

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
        if has_acc:
            # append derivative calculations
            full["score"] = last["score"]
    else:
        full = last

    # calculate loglikelihood
    full["loglik"] = last["loglik"] - n_obs * jnp.log(n_particles)
    if has_acc:
        prob = lwgt_to_prob(last["logw"])
        if not fisher:
            # calculate score
            alpha = jax.vmap(
                jnp.multiply
            )(prob, full["score"])
            full["score"] = jnp.sum(alpha, axis=0)
        else:
            # calculate score and fisher information
            alpha, gamma = jax.vmap(
                lambda p, a, b: (p * a, p * (jnp.outer(a, a) + b))
            )(prob, full["score"][0], full["score"][1])
            alpha = jnp.sum(alpha, axis=0)
            hess = jnp.sum(gamma, axis=0) - jnp.outer(alpha, alpha)
            full["score"] = alpha
            full["fisher"] = hess
    return full


def particle_filter_rb(model, key, y_meas, theta, n_particles,
                       resampler=resample_multinomial,
                       score=True, fisher=False, history=False,
                       tilde_for=False):
    r"""
    Rao-Blackwellized particle filter.

    Notes:

        - Algorithm 2 of Poyiadjis et al 2011.

        - Need to move `tilde_for` to a separate function `particle_filter_rb_for`.

    Args:
        model: Object specifying the state-space model.  It must have the following methods:
            - `state_lpdf()`
            - `meas_lpdf()`
            - `pf_step()`
            - `prop_lpdf(x_curr, x_prev, y_curr, theta)`: log-density of the proposal distribution.
        key: PRNG key.
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        n_particles: Number of particles.
        resampler: Resampling function.  Argument signature is `resampler(key, x_particles_prev, logw) => (x_particles_curr, ancestors)`.
        score: Whether or not to return an estimate of the score function at `theta`.
        fisher: Whether or not to return an estimate of the Fisher information at `theta`.  If `True` returns score as well.
        history: Whether to output the history of the filter or only the last step.

    Returns:
        A dictionary with elements:
            - `loglik`: The particle filter loglikelihood evaluated at `theta`.
            - `score`: A vector of size `n_theta = length(theta)` containing the estimated score at `theta`.
            - `fisher`: An array of size `(n_theta, n_theta)` containing the estimated observed fisher information at `theta`.
            - `x_particles`: A jax array containing the state variable particles at the last time point (leading dimension `n_particles`) or at all time points (leading dimensions `(n_obs, n_particles)` if `history=True`.
            - `logw`: A jax array containing unnormalized log weights at the last time point (dimensions `n_particles`) or at all time points (dimensions (n_obs, n_particles)`) if `history=True`.
            - `ancestors`: If `history=True`, an array of shape `(n_obs-1, n_particles)` containing the ancestor of each particle at times `t=1,...,T`.
    """
    n_obs = y_meas.shape[0]
    has_acc = score or fisher

    # internal functions for vmap
    def accumulator(x_prev, x_curr, y_curr, acc_prev, logw_aux):
        """
        Accumulator for weights and possibly derivatives calculation.

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
        logw_prop = model.prop_lpdf(x_curr=x_curr, x_prev=x_prev,
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
        return model.prop_lpdf(x_curr=x_curr, x_prev=x_prev, y_curr=y_curr,
                               theta=theta)

    def pf_step(key, x_prev, y_curr):
        """
        Sample from the proposal distribution `x_curr ~ q(x_t | x_t-1, y_t, theta)`.

        If `model.pf_prop()` is missing, use `model.pf_step()` instead.  However, in this case discards the log-weight for the proposal as it is not used in this particle filter.
        """
        if callable(getattr(model, "pf_prop", None)):
            x_curr = model.pf_prop(key=key, x_prev=x_prev, y_curr=y_curr,
                                   theta=theta)
        else:
            x_curr, _ = model.pf_step(key=key, x_prev=x_prev, y_curr=y_curr,
                                      theta=theta)
        return x_curr

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

    def pf_bar_for(x_prev, x_curr, y_curr, acc_prev, logw_aux):
        """
        Update calculations relating to logw_bar.

        This is the for-loop version, which does both loops inside the helper function.

        Returns:
            Dictionary with elements:
            - logw_bar: rao-blackwellized weights.
            - alpha: optional current value of alpha.
            - beta: optional current value of beta.
        """
        n_theta = theta.size
        # grad and hess functions
        grad_meas = jax.grad(model.meas_lpdf, argnums=2)
        grad_state = jax.grad(model.state_lpdf, argnums=2)
        hess_meas = jax.jacfwd(jax.jacrev(model.meas_lpdf, argnums=2),
                               argnums=2)
        hess_state = jax.jacfwd(jax.jacrev(model.state_lpdf, argnums=2),
                                argnums=2)
        # storage
        logw_bar = jnp.zeros(n_particles)
        _logw_targ = jnp.zeros(n_particles)
        _logw_prop = jnp.zeros(n_particles)
        alpha_curr = jnp.zeros((n_particles, n_theta))
        _alpha_full = jnp.zeros((n_particles, n_theta))
        beta_curr = jnp.zeros((n_particles, n_theta, n_theta))
        _beta_full = jnp.zeros((n_particles, n_theta, n_theta))
        for i in jnp.arange(x_curr.shape[0]):
            for j in jnp.arange(x_prev.shape[0]):
                _logw_targ = _logw_targ.at[j].set(
                    model.state_lpdf(
                        x_prev=x_prev[j],
                        x_curr=x_curr[i],
                        theta=theta
                    ) + acc_prev["logw_bar"][j]
                )
                # id_print(x_prev[j])
                # id_print(acc_prev["logw_bar"][j])
                # id_print(_logw_targ[j])
                _logw_prop = _logw_prop.at[j].set(
                    model.prop_lpdf(
                        x_prev=x_prev[j],
                        x_curr=x_curr[i],
                        y_curr=y_curr,
                        theta=theta
                    ) + logw_aux[j]
                )
                if has_acc:
                    _alpha_full = _alpha_full.at[j].set(
                        grad_meas(y_curr, x_curr[i], theta) +
                        grad_state(x_curr[i], x_prev[j], theta) +
                        acc_prev["alpha"][j]
                    )
                    # id_print(_alpha_full[j])
                    if fisher:
                        _beta_full = _beta_full.at[j].set(
                            jnp.outer(_alpha_full[j], _alpha_full[j]) +
                            hess_meas(y_curr, x_curr[i], theta) +
                            hess_state(x_curr[i], x_prev[j], theta) +
                            acc_prev["beta"][j]
                        )
            logw_tilde = jsp.special.logsumexp(_logw_targ) + \
                model.meas_lpdf(
                y_curr=y_curr,
                x_curr=x_curr[i],
                theta=theta
            )
            logw_bar = logw_bar.at[i].set(
                logw_tilde - jsp.special.logsumexp(_logw_prop)
            )
            if has_acc:
                alpha_curr = alpha_curr.at[i].set(
                    tree_mean(_alpha_full, _logw_targ)
                )
                if fisher:
                    beta_curr = beta_curr.at[i].set(
                        tree_mean(_beta_full, _logw_targ) -
                        jnp.outer(alpha_curr[i], alpha_curr[i])
                    )
        acc_curr = {"logw_bar": logw_bar}
        if has_acc:
            acc_curr["alpha"] = alpha_curr
            if fisher:
                acc_curr["beta"] = beta_curr
        return acc_curr

    # lax.scan stepping function
    def filter_step(carry, t):
        # 1. sample particles from previous time point
        key, subkey = random.split(carry["key"])
        # augment previous weights with auxillary weights.
        logw_aux = jax.vmap(
            pf_aux,
            in_axes=(0, 0, None)
        )(carry["logw_bar"], carry["x_particles"], y_meas[t])
        # resampled particles
        res_particles = resampler(
            key=subkey,
            x_particles_prev=carry["x_particles"],
            logw=logw_aux
        )
        # 2. sample particles from current timepoint
        key, *subkeys = random.split(key, num=n_particles+1)
        x_particles = jax.vmap(
            pf_step,
            in_axes=(0, 0, None)
        )(jnp.array(subkeys), res_particles["x_particles"], y_meas[t])
        # 3. compute all double summations
        acc_prev = {"logw_bar": carry["logw_bar"]}
        if has_acc:
            acc_prev["alpha"] = carry["alpha"]
            if fisher:
                acc_prev["beta"] = carry["beta"]
        if not tilde_for:
            acc_curr = jax.vmap(
                pf_bar,
                in_axes=(None, 0, None, None, None)
            )(carry["x_particles"], x_particles, y_meas[t],
              acc_prev, logw_aux)
        else:
            acc_curr = pf_bar_for(
                x_prev=carry["x_particles"],
                x_curr=x_particles,
                y_curr=y_meas[t],
                acc_prev=acc_prev,
                logw_aux=logw_aux
            )
        # output
        res_carry = {
            "x_particles": x_particles,
            "loglik": carry["loglik"] +
            jsp.special.logsumexp(acc_curr["logw_bar"]),
            "key": key,
            "resample_out": rm_keys(res_particles, "x_particles")
        }
        res_carry.update(acc_curr)
        if history:
            res_stack = rm_keys(res_carry, ["key", "loglik"])
        else:
            res_stack = None
        return res_carry, res_stack

    # lax.scan initial value
    key, *subkeys = random.split(key, num=n_particles+1)
    # initial particles and weights
    x_particles, logw = jax.vmap(
        pf_init
    )(jnp.array(subkeys))
    # dummy initialization for resampler
    init_res = resampler(key, x_particles, logw)
    init_res = tree_zeros(rm_keys(init_res, ["x_particles"]))
    filter_init = {
        "x_particles": x_particles,
        "loglik": jsp.special.logsumexp(logw),
        "logw_bar": logw,
        "key": key,
        "resample_out": init_res
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
        logw_bar = last["logw_bar"]
        if not fisher:
            # calculate score only
            full["score"] = tree_mean(last["alpha"], logw_bar)
        else:
            # calculate score and fisher information
            prob = lwgt_to_prob(logw_bar)
            alpha = last["alpha"]
            beta = last["beta"]
            alpha, gamma = jax.vmap(
                lambda p, a, b: (p * a, p * (jnp.outer(a, a) + b))
            )(prob, alpha, beta)
            alpha = jnp.sum(alpha, axis=0)
            hess = jnp.sum(gamma, axis=0) - jnp.outer(alpha, alpha)
            full["score"] = alpha
            full["fisher"] = hess

    return full
