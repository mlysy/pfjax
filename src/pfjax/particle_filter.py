"""
<<<<<<< HEAD
Particle filter in JAX.
The API requires the user to define a model class with the following methods:
- `pf_init()`
- `pf_step()`
The provided functions are:
- `particle_filter()`
- `particle_loglik()`
- `particle_smooth()`
- `particle_resample()`
=======
Particle filters which approximate the score and fisher information.

The API requires the user to define a model class with the following methods:

- `pf_init: (key, y_init, theta) => (x_particles, logw)`: Sampling and log-weights for the initial latent variable.

- `pf_step: (key, x_prev, y_curr, theta) => (x_particles, logw)`: Sampling and log-weights for each subsequent latent variable.

- `pf_aux: (x_prev, y_curr, theta) => logw`: The look-forward log-weights of the auxillary particle filter.

- `state_lpdf: (x_curr, x_prev, theta) => lpdf`: The log-density of the state model.

- `meas_lpdf: (y_curr, x_curr, theta) => lpdf`: The log-density of the measurement model.

For now the resampling function is just the multinomial, but we'll keep the `resampler` argument to eventually pass in other resamplers with `ancestors`.
>>>>>>> 52cd7e65316551e5e13b108c6b64d93ac6d01993
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from jax import random
from jax import lax
<<<<<<< HEAD
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
=======
# from jax.experimental.host_callback import id_print
from .utils import *
from .particle_resamplers import resample_multinomial
>>>>>>> 52cd7e65316551e5e13b108c6b64d93ac6d01993


def particle_filter(model, key, y_meas, theta, n_particles,
                    resampler=resample_multinomial,
                    score=False, fisher=False, history=False):
    r"""
<<<<<<< HEAD
    Apply particle filter for given value of `theta`.
    Closely follows Algorithm 2 of Murray 2013 <https://arxiv.org/abs/1306.3277>.
=======
    Basic particle filter.

    **Notes:**

    - Can optionally use an auxiliary particle filter if `model.pf_aux()` is provided.

    - The score and fisher information are estimated using the method described by Poyiadjis et al 2011, Algorithm 1. 

    - Should have the option of adding data as we go along.  So for example, could have an argument `init=None`, which if not none is the carry from `lax.scan()`.  Should then also return the carry as an output...

>>>>>>> 52cd7e65316551e5e13b108c6b64d93ac6d01993
    Args:
        model: Object specifying the state-space model having the following methods:
            - `pf_init()`
            - `pf_step()`
            - Optionally `pf_aux()`
            - Optionally `state_lpdf()` and `meas_lpdf()`, if `score or fisher == True`.
        key: PRNG key.
        y_meas: JAX array with leading dimension `n_obs` containing the measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        n_particles: Number of particles.
<<<<<<< HEAD
        resampler: Function used at step `t` to obtain sample of particles from `p(x_{t-1} | y_{0:t-1}, theta)`.  The inputs to the function are `resampler(x_particles, logw, key)`, and the return value is a dictionary with mandatory element `x_particles` and optional elements that get stacked to the final output using `lax.scan()`.  Default value is `resample_multinomial()`.
=======
        resampler: Function used at step `t` to obtain sample of particles from `p(x_{t} | y_{0:t}, theta)` out of a sample of particles from `p(x_{t-1} | y_{0:t-1}, theta)`.   The argument signature is `resampler(x_particles, logw, key)`, and the return value is a dictionary with mandatory element `x_particles`  and optional elements that get carried to the next step `t+1` via `lax.scan()`.
        score: Whether or not to return an estimate of the score function at `theta`.  Only works if `resampler` has an output element named `ancestors`.
        fisher: Whether or not to return an estimate of the Fisher information at `theta`.  Only works if `resampler` has an output element named `ancestors`.  If `True` returns score as well.
        history: Whether to output the history of the filter or only the last step.

>>>>>>> 52cd7e65316551e5e13b108c6b64d93ac6d01993
    Returns:
        A dictionary with elements:
            - `x_particles`: JAX array containing the state variable particles at the last time point (leading dimension `n_particles`) or at all time points (leading dimensions `(n_obs, n_particles)` if `history=True`.
            - `logw`: JAX array containing unnormalized log weights at the last time point (dimensions `n_particles`) or at all time points (dimensions (n_obs, n_particles)`) if `history=True`.
            - `loglik`: The particle filter loglikelihood evaluated at `theta`.
            - `score`: Optional 1D JAX array of size `n_theta = length(theta)` containing the estimated score at `theta`.
            - `fisher`: Optional JAX array of shape `(n_theta, n_theta)` containing the estimated observed fisher information at `theta`.
            - `resample_out`: If `history=True`, a dictionary of additional outputs from `resampler` function.  The leading dimension of each element of the dictionary has leading dimension `n_obs-1`, since these additional outputs do not apply to the first time point `t=0`.
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
            prob = lwgt_to_prob(last["logw"])
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


def particle_filter_rb(model, key, y_meas, theta, n_particles,
                       resampler=resample_multinomial,
                       score=False, fisher=False, history=False):
    r"""
<<<<<<< HEAD
    Calculate particle filter marginal loglikelihood.
    Args:
        logw: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.
=======
    Rao-Blackwellized particle filter.

    Notes:

        - Algorithm 2 of Poyiadjis et al 2011.

        - Can optionally use an auxiliary particle filter if `model.pf_aux()` is provided.

    Args:
        model: Object specifying the state-space model having the following methods:
            - `pf_init()`.
            - Either `pf_prop()` or `pf_step()`.
            - `state_lpdf()`.
            - `meas_lpdf()`.
            - `prop_lpdf()`.
            - Optionally `pf_aux()`.
        key: PRNG key.
        y_meas: JAX array with leading dimension `n_obs` containing the measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        n_particles: Number of particles.
        resampler: Function used at step `t` to obtain sample of particles from `p(x_{t} | y_{0:t}, theta)` out of a sample of particles from `p(x_{t-1} | y_{0:t-1}, theta)`.   The argument signature is `resampler(x_particles, logw, key)`, and the return value is a dictionary with mandatory element `x_particles`  and optional elements that get carried to the next step `t+1` via `lax.scan()`.
        score: Whether or not to return an estimate of the score function at `theta`.
        fisher: Whether or not to return an estimate of the Fisher information at `theta`.  If `True` returns score as well.
        history: Whether to output the history of the filter or only the last step.

>>>>>>> 52cd7e65316551e5e13b108c6b64d93ac6d01993
    Returns:
        A dictionary with elements:
            - `x_particles`: JAX array containing the state variable particles at the last time point (leading dimension `n_particles`) or at all time points (leading dimensions `(n_obs, n_particles)` if `history=True`.
            - `logw_bar`: JAX array containing unnormalized log weights at the last time point (dimensions `n_particles`) or at all time points (dimensions (n_obs, n_particles)`) if `history=True`.
            - `loglik`: The particle filter loglikelihood evaluated at `theta`.
            - `score`: Optional 1D JAX array of size `n_theta = length(theta)` containing the estimated score at `theta`.
            - `fisher`: Optional JAX array of shape `(n_theta, n_theta)` containing the estimated observed fisher information at `theta`.
            - `resample_out`: If `history=True`, a dictionary of additional outputs from `resampler` function.  The leading dimension of each element of the dictionary has leading dimension `n_obs-1`, since these additional outputs do not apply to the first time point `t=0`.
    """
    n_obs = y_meas.shape[0]
    has_acc = score or fisher

    # internal functions for vmap
    def accumulator(x_prev, x_curr, y_curr, acc_prev, logw_aux):
        """
        Accumulator for weights and possibly derivative calculations.

<<<<<<< HEAD
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
=======
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
>>>>>>> 52cd7e65316551e5e13b108c6b64d93ac6d01993
    }
    if has_acc:
        # dummy initialization for derivatives
        filter_init["alpha"] = jnp.zeros((n_particles, theta.size))
        if fisher:
            filter_init["beta"] = jnp.zeros(
                (n_particles, theta.size, theta.size)
            )

    # lax.scan itself
<<<<<<< HEAD
    last, full = lax.scan(fun, init, jnp.flip(jnp.arange(n_obs-1)))
    # particle indices in forward order
    i_part = jnp.flip(jnp.append(init["i_part"], full["i_part"]))
    return x_particles[jnp.arange(n_obs), i_part, ...]  # , i_part
=======
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
            prob = lwgt_to_prob(last["logw_bar"])
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
>>>>>>> 52cd7e65316551e5e13b108c6b64d93ac6d01993
