import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from jax import random
from jax import lax
# from jax.experimental.host_callback import id_print
from ..utils import *
from ..particle_filter import resample_multinomial


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
            return tree_add(tree1=acc_prev, tree2=acc_curr)

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
            "resample_out": rm_keys(new_particles, ["x_particles", "logw"])
        }
        if has_acc:
            res_carry["accumulate_out"] = acc_curr
        res_stack = rm_keys(res_carry, ["key", "loglik"]) if history else None
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
    init_resample = rm_keys(init_resample, ["x_particles", "logw"])
    init_resample = tree_zeros(init_resample)
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
            full["accumulate_out"] = tree_mean(
                tree=full["accumulate_out"],
                logw=full["logw"]
            )
    # calculate loglikelihood
    full["loglik"] = last["loglik"] - n_obs * jnp.log(n_particles)
    return full
