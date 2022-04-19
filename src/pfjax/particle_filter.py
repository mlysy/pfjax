"""
Particle filter in JAX.

The API requires the user to define a model class with the following methods:

- `pf_init()`
- `pf_step()`

The provided functions are:
- `particle_filter()`
- `particle_loglik()`
- `particle_smooth()`
- `particle_resample()`

"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from jax import random
from jax import lax
# from jax.experimental.maps import xmap
import ott
from ott.geometry import pointcloud
from ott.core import sinkhorn


def lwgt_to_prob(logw):
    r"""
    Calculate normalized probabilities from unnormalized log weights.

    Args:
        logw: Vector of `n_particles` unnormalized log-weights.

    Returns:
        Vector of `n_particles` normalized weights that sum to 1.
    """
    wgt = jnp.exp(logw - jnp.max(logw))
    prob = wgt / jnp.sum(wgt)
    return prob


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


def resample_multinomial(key, x_particles_prev, logw):
    r"""
    Particle resampler.

    This basic one just does a multinomial sampler, i.e., sample with replacement proportional to weights.

    Args:
        key: PRNG key.
        x_particles_prev: An `ndarray` with leading dimension `n_particles` consisting of the particles from the previous time step.
        logw: Vector of corresponding `n_particles` unnormalized log-weights.

    Returns:
        A dictionary with elements:
            - `x_particles`: An `ndarray` with leading dimension `n_particles` consisting of the particles from the current time step.  These are sampled with replacement from `x_particles_prev` with probability vector `exp(logw) / sum(exp(logw))`.
            - `ancestors`: Vector of `n_particles` integers between 0 and `n_particles-1` giving the index of each element of `x_particles_prev` corresponding to the elements of `x_particles`.
    """
    prob = lwgt_to_prob(logw)
    n_particles = logw.size
    ancestors = random.choice(key,
                              a=jnp.arange(n_particles),
                              shape=(n_particles,), p=prob)
    return {
        "x_particles": x_particles_prev[ancestors, ...],
        "ancestors": ancestors
    }


def resample_mvn(key, x_particles_prev, logw):
    r"""
    Particle resampler with Multivariate Normal approximation.

    Args:
        key: PRNG key.
        x_particles_prev: An `ndarray` with leading dimension `n_particles` consisting of the particles from the previous time step.
        logw: Vector of corresponding `n_particles` unnormalized log-weights.

    Returns:
        A dictionary with elements:
            - `x_particles`: An `ndarray` with leading dimension `n_particles` consisting of the particles from the current time step.
            - `mvn_mean`: Vector of length `n_state = prod(x_particles.shape[1:])` representing the mean of the MVN.
            - `mvn_cov`: Matrix of size `n_state x n_state` representing the covariance matrix of the MVN.
    """
    prob = lwgt_to_prob(logw)
    p_shape = x_particles_prev.shape
    n_particles = p_shape[0]
    # calculate weighted mean and variance
    x_particles = jnp.transpose(x_particles_prev.reshape((n_particles, -1)))
    mvn_mean = jnp.average(x_particles, axis=1, weights=prob)
    mvn_cov = jnp.atleast_2d(jnp.cov(x_particles, aweights=prob))
    # for numeric stability
    mvn_cov += jnp.diag(jnp.ones(mvn_cov.shape[0]) * 1e-10)
    x_particles = random.multivariate_normal(key,
                                             mean=mvn_mean,
                                             cov=mvn_cov,
                                             shape=(n_particles,))
    return {
        "x_particles": jnp.reshape(x_particles, newshape=p_shape),
        "mvn_mean": mvn_mean,
        "mvn_cov": mvn_cov
    }


def resample_ot(key, x_particles_prev, logw,
                pointcloud_kwargs={},
                sinkhorn_kwargs={}):
    r"""
    Particle resampler using optimal transport.

    Based on Algorithms 2 and 3 of Corenflos et al 2021 <https://arxiv.org/abs/2102.07850>.

    **Notes:**

    - Argument `jit` to `ott.sinkhorn.sinkhorn()` is ignored, i.e., always set to `False`.

    Args:
        key: PRNG key.
        x_particles_prev: An `ndarray` with leading dimension `n_particles` consisting of the particles from the previous time step.
        logw: Vector of corresponding `n_particles` unnormalized log-weights.
        pointcloud_kwargs: Dictionary of additional arguments to `ott.pointcloud.PointCloud()`.
        sinkhorn_kwargs: Dictionary of additional arguments to `ott.sinkhorn.sinkhorn()`.

    Returns:
        A dictionary with elements:
            - `x_particles`: An `ndarray` with leading dimension `n_particles` consisting of the particles from the current time step.
            - `geom`: An `ott.Geometry` object.
            - `sink`: The output of the call to `ott.sinkhorn.sinkhorn()`.
    """
    sinkhorn_kwargs.update(jit=False)
    prob = lwgt_to_prob(logw)
    p_shape = x_particles_prev.shape
    n_particles = p_shape[0]
    x_particles = x_particles_prev.reshape((n_particles, -1))
    geom = pointcloud.PointCloud(x=x_particles, y=x_particles,
                                 **pointcloud_kwargs)
    sink = sinkhorn.sinkhorn(geom,
                             a=prob,
                             b=jnp.ones(n_particles),
                             **sinkhorn_kwargs)
    x_particles = geom.apply_transport_from_potentials(
        f=sink.f, g=sink.g, vec=x_particles.T
    )
    return {
        "x_particles": jnp.reshape(x_particles.T, newshape=p_shape),
        "geom": geom,
        "sink": sink
    }


def particle_filter(model, key, y_meas, theta, n_particles,
                    resampler=resample_multinomial):
    r"""
    Apply particle filter for given value of `theta`.

    Closely follows Algorithm 2 of Murray 2013 <https://arxiv.org/abs/1306.3277>.

    Args:
        model: Object specifying the state-space model.
        key: PRNG key.
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        n_particles: Number of particles.
        resampler: Function used at step `t` to obtain sample of particles from `p(x_{t-1} | y_{0:t-1}, theta)`.  The inputs to the function are `resampler(x_particles, logw, key)`, and the return value is a dictionary with mandatory element `x_particles` and optional elements that get stacked to the final output using `lax.scan()`.  Default value is `resample_multinomial()`.

    Returns:
        A dictionary with elements:
            - `x_particles`: An `ndarray` with leading dimensions `(n_obs, n_particles)` containing the state variable particles.
            - `logw`: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.
            - `...`: Other `ndarray`s with leading dimension `n_obs-1`, corresponding to additional outputs from `resampler()` as accumulated by `lax.scan()`.  Since these additional outputs do not apply to the first time step (since it has no previous time step), the leading dimension of each additional output is `n_obs-1`.
    """
    n_obs = y_meas.shape[0]

    # lax.scan setup
    # scan function
    def fun(carry, t):
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
            lambda xs, k: model.pf_step(k, xs, y_meas[t], theta)
        )(new_particles["x_particles"], jnp.array(subkeys))
        # output
        res_carry = {
            "x_particles": x_particles,
            "logw": logw,
            "key": key
        }
        res_stack = new_particles
        res_stack["x_particles"] = x_particles
        res_stack["logw"] = logw
        return res_carry, res_stack
    # scan initial value
    key, *subkeys = random.split(key, num=n_particles+1)
    # vmap version
    x_particles, logw = jax.vmap(
        lambda k: model.pf_init(k, y_meas[0], theta))(jnp.array(subkeys))
    # xmap version: experimental!
    # x_particles = xmap(
    #     lambda ym, th, k: model.init_sample(ym, th, k),
    #     in_axes=([...], [...], ["particles", ...]),
    #     out_axes=["particles", ...])(y_meas[0], theta, jnp.array(subkeys))
    # logw = xmap(
    #     lambda xs, ym, th: model.init_logw(xs, ym, th),
    #     in_axes=(["particles", ...], [...], [...]),
    #     out_axes=["particles", ...])(x_particles, y_meas[0], theta)
    init = {
        "x_particles": x_particles,
        "logw": logw,
        "key": key
    }
    resampler(key=init["key"], x_particles_prev=init["x_particles"],
              logw=init["logw"])
    # lax.scan itself
    last, full = lax.scan(fun, init, jnp.arange(1, n_obs))
    # append initial values of x_particles and logw
    full["x_particles"] = jnp.append(
        jnp.expand_dims(init["x_particles"], axis=0),
        full["x_particles"], axis=0)
    full["logw"] = jnp.append(
        jnp.expand_dims(init["logw"], axis=0),
        full["logw"], axis=0)
    return full


def particle_loglik(logw):
    r"""
    Calculate particle filter marginal loglikelihood.

    Args:
        logw: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.

    Returns:
        Particle filter approximation of
        ```
        log p(y_meas | theta) = log int p(y_meas | x_state, theta) * p(x_state | theta) dx_state
        ```
    """
    n_particles = logw.shape[1]
    return jnp.sum(jsp.special.logsumexp(logw, axis=1) - jnp.log(n_particles))


def particle_smooth(key, logw, x_particles, ancestors):
    r"""
    Draw a sample from `p(x_state | x_meas, theta)` using the basic particle smoothing algorithm.

    **FIXME:**

    - Will probably need to change inputs to "generalize" to other resampling methods.

    Args:
        key: PRNG key.
        logw: Vector of `n_particles` unnormalized log-weights at the last time point `t = n_obs-1`.
        x_particles: An `ndarray` with leading dimensions `(n_obs, n_particles)` containing the state variable particles.
        ancestors: An integer `ndarray` of shape `(n_obs, n_particles)` where each element gives the index of the particle's ancestor at the previous time point.

    Returns:
        An `ndarray` with leading dimension `n_obs` sampled from `p(x_{0:T} | y_{0:T}, theta)`.
    """
    n_particles = logw.size
    n_obs = x_particles.shape[0]
    prob = lwgt_to_prob(logw)
    # wgt = jnp.exp(logw - jnp.max(logw))
    # prob = wgt / jnp.sum(wgt)

    # lax.scan setup
    # scan function
    def fun(carry, t):
        # ancestor particle index
        i_part = ancestors[t, carry["i_part"]]
        res = {"i_part": i_part}
        return res, res
        # res_carry = {"i_part": i_part}
        # res_stack = {"i_part": i_part, "x_state": x_particles[t, i_part]}
        # return res_carry, res_stack
    # scan initial value
    init = {
        "i_part": random.choice(key, a=jnp.arange(n_particles), p=prob)
    }
    # lax.scan itself
    last, full = lax.scan(fun, init, jnp.flip(jnp.arange(n_obs-1)))
    # particle indices in forward order
    i_part = jnp.flip(jnp.append(init["i_part"], full["i_part"]))
    return x_particles[jnp.arange(n_obs), i_part, ...]  # , i_part


# def _accumulate_none(x_prev, x_curr, y_curr, theta):
#     """
#     Empty accumulator.
#     """
#     return None


def particle_filter2(model, key, y_meas, theta, n_particles,
                     resampler=resample_multinomial,
                     history=False,
                     accumulator=None):
    r"""
    Apply particle filter for given value of `theta`.

    Closely follows Algorithm 2 of Murray 2013 <https://arxiv.org/abs/1306.3277>.

    Notes:

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
