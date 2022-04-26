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
from ..particle_filter import lwgt_to_prob, resample_multinomial

# --- tree helper functions ----------------------------------------------------


def _tree_add(tree1, tree2):
    r"""
    Add two pytrees leafwise.
    """
    return jtu.tree_map(lambda x, y: x+y, tree1, tree2)
    # return jtu.tree_map(lambda x, y: x+y[0], tree1, (tree2,))


def _tree_mean(tree, logw):
    r"""
    Weighted mean of each leaf of a pytree along leading dimension.
    """
    prob = lwgt_to_prob(logw)
    broad_mult = jtu.Partial(jax.vmap(jnp.multiply), prob)
    return jtu.tree_map(
        jtu.Partial(jnp.sum, axis=0),
        jtu.tree_map(broad_mult, tree)
    )


def _tree_shuffle(tree, index):
    """
    Shuffle the leading dimension of each leaf of a pytree by values in index.
    """
    return jtu.tree_map(lambda x: x[index, ...], tree)


def _tree_zeros(tree):
    r"""
    Fill pytree with zeros.
    """
    return jtu.tree_map(lambda x: jnp.zeros_like(x), tree)


def _rm_keys(x, keys):
    r"""
    Remove specified keys from given dict.
    """
    return {k: x[k] for k in x.keys() if k not in keys}


# --- particle filters ---------------------------------------------------------

def particle_accumulator(model, key, y_meas, theta, n_particles,
                         resampler=resample_multinomial,
                         history=False,
                         accumulator=None):
    r"""
    Particle filter with accumulator.

    Notes:

    - This is a precursor to `auxillary_filter_fast()` with a more general accumulator.  It will probably be depreciated however since the accumulator has very high variance and is thus not recommended to use in practice.

    - **Warning:** The accumulator only works with `resampler = resample_multinomial()`.

    - May wish to remove `resample_out` when `resampler()` has no additional outputs.

    - `resampler()` could return additional outputs more conveniently, e.g., as a single additional key `resample_out` consisting of a pytree.  However, this isn't backwards compatible with `particle_filter()` so haven't implemented it yet.

    - Accumulator is initialized with a pytree of zeros.  This precludes accumulating something different for time `t=0`, e.g., full score estimation.  However, the contribution from time 0 is often negligible.  Also, with some extra work it's probably possible to account for it using more code...

    Args:
        model: Object specifying the state-space model.
        key: PRNG key.
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        n_particles: Number of particles.
        resampler: Function used at step `t` to obtain sample of particles from `p(x_{t-1} | y_{0:t-1}, theta)`.  The inputs to the function are `resampler(x_particles, logw, key)`, and the return value is a dictionary with mandatory element `x_particles` and optional elements that get stacked to the final output using `lax.scan()`.  Default value is `resample_multinomial()`.
        history: Whether to output the entire history of the filter or only the last step.
        accumulator: Function `phi(x_{t-1}, x_t, y_t, theta)` with arguments `x_prev`, `x_curr`, `y_curr`, `theta` and outputs a JAX pytree such that the particle filter will approximate

            ```
            E[sum(phi(x_{t-1}, x_t, y_t, theta)) | y_{0:T}]
            ```

    Returns:
        A dictionary with elements:
            - `loglik`: The particle filter loglikelihood evaluated at `theta`.
            - `x_particles`: A jax array containing the state variable particles at the last time point (leading dimension `n_particles`) or at all time points (leading dimensions `(n_obs, n_particles)` if `history=True`.
            - `logw`: A jax array containing unnormalized log weights at the last time point (dimensions `n_particles`) or at all time points (dimensions (n_obs, n_particles)`) if `history=True`.
            - `resample_out`: Jax pytree corresponding to additional outputs from `resampler()` as accumulated by `lax.scan()`.  Either for the last time point if `history=False`, or for all timepoints if `history=True`, in which case the leading dimension in each leaf of the pytree is `n_obs-1` since these additional outputs do not apply to the first time point.
           - `accumulate_out`: Optional Jax pytree corresponding to the estimate of the expectation defined by the `accumulator` function.  If `history=True` the leading dimension of each leaf of the pytree is `(n_obs-1, n_particles)` and summation is not performed.  In other words, in `leaf[i,j,...]` of the returned pytree we have `accumulator(x_particles[i-1,j], x_particles[i,j], y_meas[i], theta)`.
    """
    n_obs = y_meas.shape[0]
    has_acc = accumulator is not None

    # internal functions for vectorizing
    def pf_step(key, x_prev, y_curr):
        return model.pf_step(key=key, x_prev=x_prev, y_curr=y_curr, theta=theta)

    def pf_init(key):
        return model.pf_init(key=key, y_init=y_meas[0], theta=theta)

    def pf_acc(acc_prev, x_prev, x_curr, y_curr):
        acc_curr = accumulator(
            x_prev=x_prev, x_curr=x_curr, y_curr=y_curr, theta=theta
        )
        if history:
            return acc_curr
        else:
            return _tree_add(tree1=acc_prev, tree2=acc_curr)

    # lax.scan setup
    # scan function
    def filter_step(carry, t):
        # sample particles from previous time point
        key, subkey = random.split(carry["key"])
        new_particles = resampler(
            key=subkey,
            x_particles_prev=carry["x_particles"],
            logw=carry["logw"]
        )
        # update particles to current time point (and get weights)
        key, *subkeys = random.split(key, num=n_particles+1)
        x_particles, logw = jax.vmap(
            pf_step,
            in_axes=(0, 0, None)
        )(jnp.array(subkeys), new_particles["x_particles"], y_meas[t])
        if has_acc:
            # accumulate expectation
            acc_prev = carry["accumulate_out"]
            if not history:
                # resample acc_prev
                acc_prev = _tree_shuffle(
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
            "resample_out": _rm_keys(new_particles, ["x_particles", "logw"])
        }
        if has_acc:
            res_carry["accumulate_out"] = acc_curr
        res_stack = _rm_keys(res_carry, ["key", "loglik"]) if history else None
        return res_carry, res_stack
    # scan initial value
    key, *subkeys = random.split(key, num=n_particles+1)
    # vmap version
    x_particles, logw = jax.vmap(
        # lambda k: model.pf_init(k, y_meas[0], theta)
        pf_init
    )(jnp.array(subkeys))
    # dummy initialization for resample
    init_resample = resampler(key, x_particles, logw)
    init_resample = _rm_keys(init_resample, ["x_particles", "logw"])
    init_resample = _tree_zeros(init_resample)
    if has_acc:
        # dummy initialization for accumulate
        init_acc = jax.vmap(
            accumulator,
            in_axes=(0, 0, None, None)
        )(x_particles, x_particles, y_meas[0], theta)
        init_acc = _tree_zeros(init_acc)
    filter_init = {
        "x_particles": x_particles,
        "logw": logw,
        "loglik": jsp.special.logsumexp(logw),
        "key": key,
        "resample_out": init_resample
    }
    if has_acc:
        filter_init["accumulate_out"] = init_acc
    # lax.scan itself
    last, full = lax.scan(filter_step, filter_init, jnp.arange(1, n_obs))
    if history:
        # append initial values of x_particles and logw
        full["x_particles"] = jnp.concatenate([
            filter_init["x_particles"][None], full["x_particles"]
        ])
        full["logw"] = jnp.concatenate([
            filter_init["logw"][None], full["logw"]
        ])
    else:
        full = last
        if has_acc:
            # weighted average of accumulated values
            full["accumulate_out"] = _tree_mean(
                tree=full["accumulate_out"],
                logw=full["logw"]
            )
    # calculate loglikelihood
    full["loglik"] = last["loglik"] - n_obs * jnp.log(n_particles)
    return full


def auxillary_filter_linear(model, key, y_meas, theta, n_particles,
                            resampler=resample_multinomial,
                            score=True, fisher=False, history=False):
    r"""
    Auxillary particle filter with linear score / Fisher information calculation.

    Notes:

        - This is a slimmed-down version of `particle_accumulator()`, of which several features have been disabled as they are unlikely to be used in practice.
        - Auxillary filter currently disabled.

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
        # return model.pf_aux(x_prev=x_prev, y_curr=y_curr, theta=theta) + logw_prev
        return logw_prev

    def pf_acc(acc_prev, x_prev, x_curr, y_curr):
        acc_curr = accumulator(
            x_prev=x_prev, x_curr=x_curr, y_curr=y_curr, theta=theta
        )
        return _tree_add(tree1=acc_prev, tree2=acc_curr)

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
            acc_prev = _tree_shuffle(
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
        res_stack = _rm_keys(
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
        init_acc = _tree_zeros(init_acc)
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
                jnp.add
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


def auxillary_filter_quad(model, key, y_meas, theta, n_particles,
                          resampler=resample_multinomial,
                          score=True, fisher=False, history=False,
                          tilde_for=False):
    r"""
    Auxillary particle filter with quadratic score / Fisher information calculation.

    Notes:

        - Algorithm 2 of Poyiadjis et al 2011.

        - Auxillary filter currently disabled.

        - Loglikelihood not currently returned.

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
    def accumulator(acc_prev, x_prev, x_curr, y_curr):
        if fisher:
            alpha_prev, beta_prev = acc_prev
        else:
            alpha_prev = acc_prev
        lpdf = model.state_lpdf(x_curr=x_curr, x_prev=x_prev, theta=theta)
        grad_meas = jax.grad(model.meas_lpdf, argnums=2)
        grad_state = jax.grad(model.state_lpdf, argnums=2)
        alpha_acc = grad_meas(y_curr, x_curr, theta) + \
            grad_state(x_curr, x_prev, theta) + \
            alpha_prev
        if fisher:
            hess_meas = jax.jacfwd(jax.jacrev(model.meas_lpdf, argnums=2),
                                   argnums=2)
            hess_state = jax.jacfwd(jax.jacrev(model.state_lpdf, argnums=2),
                                    argnums=2)
            beta_acc = jnp.outer(alpha_acc, alpha_acc) + \
                hess_meas(y_curr, x_curr, theta) + \
                hess_state(x_curr, x_prev, theta) + \
                beta_prev
            return lpdf, alpha_acc, beta_acc
        else:
            return lpdf, alpha_acc

    # internal functions for vmap

    def pf_prop(key, x_prev, y_curr):
        """
        Calculate proposal `x_curr` and its log-density `logw_prop`.

        Can get this from pf_step, state_lpdf, and meas_lpdf, or directly from pf_prop.
        """
        if callable(getattr(model, "pf_prop", None)):
            x_curr, logw_prop = model.pf_prop(
                key=key,
                x_prev=x_prev,
                y_curr=y_curr,
                theta=theta
            )
            return x_curr, logw_prop
        else:
            x_curr, logw_step = model.pf_step(
                key=key,
                x_prev=x_prev,
                y_curr=y_curr,
                theta=theta
            )
            logw_targ = model.state_lpdf(
                x_curr=x_curr,
                x_prev=x_prev,
                theta=theta
            )
            logw_targ = logw_targ + model.meas_lpdf(
                y_curr=y_curr,
                x_curr=x_curr,
                theta=theta
            )
            return x_curr, logw_targ - logw_step

    def pf_init(key):
        return model.pf_init(key=key, y_init=y_meas[0], theta=theta)

    def pf_aux(logw_prev, x_prev, y_curr):
        """
        Calculate the log-density for auxillary sampling.

        If model.pf_aux is missing, taken to return 0.
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

    def pf_tilde(logw_bar, x_prev, x_curr, y_curr):
        lpdf = jax.vmap(
            model.state_lpdf,
            in_axes=(None, 0, None)
        )(x_curr, x_prev, theta)
        lpdf = jsp.special.logsumexp(logw_bar + lpdf)
        return model.meas_lpdf(
            y_curr=y_curr,
            x_curr=x_curr,
            theta=theta
        ) + lpdf

    def pf_acc(logw_bar, x_prev, x_curr, y_curr, acc_prev):
        acc = jax.vmap(
            accumulator,
            in_axes=(0, 0, None, None)
        )(acc_prev, x_prev, x_curr, y_curr)
        lpdf = acc[0]
        grad_acc = acc[1:] if fisher else acc[1]
        logw = logw_bar + lpdf
        grad_curr = _tree_mean(grad_acc, logw)
        if fisher:
            alpha_curr = grad_curr[0]
            beta_curr = grad_curr[1] - \
                jnp.outer(alpha_curr, alpha_curr)
            return alpha_curr, beta_curr
        else:
            alpha_curr = grad_curr
            return alpha_curr

    def pf_tilde_for(logw_bar, x_prev, x_curr, y_curr):
        lpdf = jnp.zeros((n_particles))
        _lpdf = jnp.zeros((n_particles))
        for i in jnp.arange(x_curr.shape[0]):
            for j in jnp.arange(x_prev.shape[0]):
                _lpdf = _lpdf.at[j].set(model.state_lpdf(
                    x_prev=x_prev[j],
                    x_curr=x_curr[i],
                    theta=theta
                ))
                # lpdf = jax.vmap(
                #     model.state_lpdf,
                #     in_axes=(None, 0, None)
                # )(x_curr, x_prev, theta)
            lpdf = lpdf.at[i].set(
                jsp.special.logsumexp(logw_bar + _lpdf) +
                model.meas_lpdf(
                    y_curr=y_curr,
                    x_curr=x_curr[i],
                    theta=theta
                )
            )
        return lpdf

    def pf_acc_for(logw_bar, x_prev, x_curr, y_curr, acc_prev):
        n_theta = theta.size
        # grad and hess
        grad_meas = jax.grad(model.meas_lpdf, argnums=2)
        grad_state = jax.grad(model.state_lpdf, argnums=2)
        hess_meas = jax.jacfwd(jax.jacrev(model.meas_lpdf, argnums=2),
                               argnums=2)
        hess_state = jax.jacfwd(jax.jacrev(model.state_lpdf, argnums=2),
                                argnums=2)
        if fisher:
            alpha_prev, beta_prev = acc_prev
        else:
            alpha_prev = acc_prev
        # storage
        _lpdf = jnp.zeros((n_particles,))
        alpha_curr = jnp.zeros((n_particles, n_theta))
        _alpha = jnp.zeros((n_particles, n_theta))
        beta_curr = jnp.zeros((n_particles, n_theta, n_theta))
        _beta = jnp.zeros((n_particles, n_theta, n_theta))
        for i in jnp.arange(x_curr.shape[0]):
            for j in jnp.arange(x_prev.shape[0]):
                _lpdf = _lpdf.at[j].set(model.state_lpdf(
                    x_prev=x_prev[j],
                    x_curr=x_curr[i],
                    theta=theta
                ))
                _alpha = _alpha.at[j].set(
                    grad_meas(y_curr, x_curr[i], theta) +
                    grad_state(x_curr[i], x_prev[j], theta) +
                    alpha_prev[j]
                )
                if fisher:
                    _beta = _beta.at[j].set(
                        jnp.outer(_alpha[j], _alpha[j]) +
                        hess_meas(y_curr, x_curr[i], theta) +
                        hess_state(x_curr[i], x_prev[j], theta) +
                        beta_prev[j]
                    )
            # prob = lwgt_to_prob(logw_bar + _lpdf)
            # alpha_curr[i] = jnp.sum(prob * _alpha)
            logw = logw_bar + _lpdf
            alpha_curr = alpha_curr.at[i].set(_tree_mean(_alpha, logw))
            if fisher:
                beta_curr = beta_curr.at[i].set(
                    _tree_mean(_beta, logw) -
                    jnp.outer(alpha_curr[i], alpha_curr[i])
                )
        if fisher:
            return alpha_curr, beta_curr
        else:
            return alpha_curr

    # lax.scan stepping function
    def filter_step(carry, t):
        # sample particles from previous time point
        key, subkey = random.split(carry["key"])
        logw_aux = jax.vmap(
            pf_aux,
            in_axes=(0, 0, None)
        )(carry["logw_bar"], carry["x_particles"], y_meas[t])
        res_particles = resampler(
            key=subkey,
            x_particles_prev=carry["x_particles"],
            logw=logw_aux
        )
        # update particles to current time point (and get weights)
        key, *subkeys = random.split(key, num=n_particles+1)
        x_particles, logw_prop = jax.vmap(
            pf_prop,  # fixme: pf_step as a function of pf_prop
            in_axes=(0, 0, None)
        )(jnp.array(subkeys), res_particles["x_particles"], y_meas[t])
        # compute logw_tilde = log p_tilde(x_t, y_0:t) and logw_bar
        if not tilde_for:
            logw_tilde = jax.vmap(
                pf_tilde,
                in_axes=(None, None, 0, None)
            )(carry["logw_bar"], res_particles["x_particles"],
              x_particles, y_meas[t])
        else:
            logw_tilde = pf_tilde_for(
                logw_bar=carry["logw_bar"],
                x_prev=res_particles["x_particles"],
                x_curr=x_particles,
                y_curr=y_meas[t]
            )
        logw_bar = logw_tilde - logw_prop
        # gradient update
        if has_acc:
            # accumulate expectation
            # note: we're calling the accumulated value "score"
            # so we don't need to rename the dictionary item later.
            if tilde_for:
                acc_curr = pf_acc_for(
                    logw_bar=carry["logw_bar"],
                    x_prev=res_particles["x_particles"],
                    x_curr=x_particles,
                    y_curr=y_meas[t],
                    acc_prev=carry["score"]
                )
            else:
                acc_curr = jax.vmap(
                    pf_acc,
                    in_axes=(None, None, 0, None, None),
                )(carry["logw_bar"], res_particles["x_particles"],
                  x_particles, y_meas[t], carry["score"])
        # output
        res_carry = {
            "x_particles": x_particles,
            # "logw_prop": logw_prop,
            # "logw_aux": logw_aux,
            "logw_bar": logw_bar,
            "logw_tilde": logw_tilde,
            "key": key,
            "ancestors": res_particles["ancestors"]
        }
        if has_acc:
            res_carry["score"] = acc_curr
        # res_stack = _rm_keys(
        #     res_carry, ["key", "loglik", "score"]
        # ) if history else None
        if history:
            res_stack = _rm_keys(res_carry, ["key", "score"])
            # res_stack["logw_prop"] = logw_prop
            # res_stack["logw_aux"] = logw_aux
            # res_stack["logw_tilde"] = logw_tilde
        else:
            res_stack = None
        return res_carry, res_stack

    # lax.scan initial value
    key, *subkeys = random.split(key, num=n_particles+1)
    # initial particles and weights
    x_particles, logw = jax.vmap(
        pf_init
    )(jnp.array(subkeys))
    # dummy initialization for ancestors
    init_ancestors = jnp.array([0] * n_particles)
    filter_init = {
        "x_particles": x_particles,
        "logw_bar": logw,
        "logw_tilde": logw,
        # "loglik": jsp.special.logsumexp(logw),
        "key": key,
        "ancestors": init_ancestors
    }
    if has_acc:
        # dummy initialization for accumulate
        alpha_init = jnp.zeros((n_particles, theta.size))
        if fisher:
            beta_init = jnp.zeros((n_particles, theta.size, theta.size))
            init_acc = (alpha_init, beta_init)
        else:
            init_acc = alpha_init
        filter_init["score"] = init_acc

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
        logw_tilde = full["logw_tilde"][-1]
        logw_bar = full["logw_bar"][-1]
        # if has_acc:
        #     if fisher:
        #         alpha = last["score"][0][-1]
        #         beta = full["score"][1][-1]
        #     else:
        #         alpha = full["score"][-1]
    else:
        full = last
        logw_tilde = full["logw_tilde"]
        logw_bar = full["logw_bar"]
        # if has_acc:
        #     if fisher:
        #         alpha = full["score"][0]
        #         beta = full["score"][1]
        #     else:
        #         alpha = full["score"]
    # calculate loglikelihood
    full["loglik"] = jsp.special.logsumexp(logw_tilde)
    if has_acc:
        prob = lwgt_to_prob(logw_bar)
        if not fisher:
            # calculate score
            alpha = last["score"]
            alpha = jax.vmap(
                jnp.add
            )(prob, alpha)
            full["score"] = jnp.sum(alpha, axis=0)
        else:
            # calculate score and fisher information
            alpha = last["score"][0]
            beta = last["score"][1]
            alpha, gamma = jax.vmap(
                lambda p, a, b: (p * a, p * (jnp.outer(a, a) + b))
            )(prob, alpha, beta)
            alpha = jnp.sum(alpha, axis=0)
            hess = jnp.sum(gamma, axis=0) - jnp.outer(alpha, alpha)
            full["score"] = alpha
            full["fisher"] = hess

    return full
