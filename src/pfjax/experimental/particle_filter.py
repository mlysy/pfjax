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


def particle_smooth(x_particles, ancestors, id_particle_last):
    """
    Return a full particle by backtracking through ancestors of particle `i_part` at last time point.

    Differs from the version in the `pfjax.particle.filter` module in that the latter does random sampling whereas here the index of the final particle is fixed.

    Args:
        x_particles: JAX array with leading dimensions `(n_obs, n_particles)` containing the state variable particles.
        ancestors: JAX integer array of shape `(n_obs-1, n_particles)` where each element gives the index of the particle's ancestor at the previous time point.
        id_particle_last: Index of the particle at the last time point `t = n_obs-1`.  An integer between `0` and `n_particles-1`.  Wrap in a JAX (scalar) array to prevent `jax.jit()` treating this as a static argument.

    Returns:
        A JAX array with leading dimension `n_obs` corresponding to the full particle having index `id_particle_last` at time `t = n_obs-1`.
    """
    n_obs = x_particles.shape[0]

    # scan function
    def get_ancestor(id_particle_next, t):
        # ancestor particle index
        id_particle_curr = ancestors[t, id_particle_next]
        return id_particle_curr, id_particle_curr

    # lax.scan
    id_particle_first, id_particle_full = \
        jax.lax.scan(get_ancestor, id_particle_last,
                     jnp.arange(n_obs-1), reverse=True)
    # append last particle index
    id_particle_full = jnp.concatenate(
        [id_particle_full, jnp.array(id_particle_last)[None]]
    )
    return x_particles[jnp.arange(n_obs), id_particle_full, ...]


def accumulate_smooth(logw, x_particles, ancestors, y_meas, theta, accumulator, mean=True):
    """
    Accumulate expectation using the basic particle smoother.

    Performs exactly the same calculation as the accumulator in `particle_accumulator()`, except by smoothing the particle history instead of directly in the filter step (no history required).

    Args:
        logw: JAX array of shape `(n_particles,)` of unnormalized log-weights at the last time point `t=n_obs-1`.
        x_particles: JAX array with leading dimensions `(n_obs, n_particles)` containing the state variable particles.
        ancestors: JAX integer array of shape `(n_obs-1, n_particles)` where each element gives the index of the particle's ancestor at the previous time point.
        y_meas: JAX array with leading dimension `n_obs` containing the measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        accumulator: Function with argument signature `(x_prev, x_curr, y_curr, theta)` returning a Pytree.  See `particle_accumulator()`.
        mean: Whether or not to compute the weighted average of the accumulated values, or to return a Pytree with each leaf having leading dimension `n_particles`.

    Returns:
        A Pytree of accumulated values.
    """
    # Get full set of particles
    n_particles = x_particles.shape[1]
    x_particles_full = jax.vmap(
        lambda i: particle_smooth(x_particles, ancestors, i)
    )(jnp.arange(n_particles))
    x_particles_prev = x_particles_full[:, :-1]
    x_particles_curr = x_particles_full[:, 1:]
    y_curr = y_meas[1:]
    acc_out = jax.vmap(
        jax.vmap(
            accumulator,
            in_axes=(0, 0, 0, None)
        ),
        in_axes=(0, 0, None, None)
    )(x_particles_prev, x_particles_curr, y_curr, theta)
    acc_out = jtu.tree_map(lambda x: jnp.sum(x, axis=1), acc_out)
    if mean:
        return _tree_mean(acc_out, logw)
    else:
        return acc_out


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


def auxillary_filter_quad(model, key, y_meas, theta, n_particles,
                          resampler=resample_multinomial,
                          score=True, fisher=False, history=False,
                          tilde_for=False):
    r"""
    Auxillary particle filter with quadratic score / Fisher information calculation.

    Notes:

        - Algorithm 2 of Poyiadjis et al 2011, except weights.  Here we use the basic weights for now, as it avoids having to define `model.prop_lpdf()`.

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
            grad_curr = _tree_mean(
                tree=_rm_keys(acc_full, ["logw_targ", "logw_prop"]),
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
                    _tree_mean(_alpha_full, _logw_targ)
                )
                if fisher:
                    beta_curr = beta_curr.at[i].set(
                        _tree_mean(_beta_full, _logw_targ) -
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
            "resample_out": _rm_keys(res_particles, "x_particles")
        }
        res_carry.update(acc_curr)
        if history:
            res_stack = _rm_keys(res_carry, ["key", "loglik"])
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
    init_res = _tree_zeros(_rm_keys(init_res, ["x_particles"]))
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
            full["score"] = _tree_mean(last["alpha"], logw_bar)
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
