"""
Utilities for both formal and interactive testing. 
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import jax.tree_util as jtu
import jax.lax as lax
from pfjax.particle_resamplers import resample_multinomial
from pfjax.utils import *
from pfjax.loglik_full import loglik_full

# --- non-exported functions for testing ---------------------------------------


def resample_multinomial_old(key, logw):
    r"""
    Particle resampler.

    This basic one just does a multinomial sampler, i.e., sample with replacement proportional to weights.

    Old API, to be depreciated after testing against `particle_filter_for()`.

    Args:
        key: PRNG key.
        logw: Vector of `n_particles` unnormalized log-weights.

    Returns:
        Vector of `n_particles` integers between 0 and `n_particles-1`, sampled with replacement with probability vector `exp(logw) / sum(exp(logw))`.
    """
    # wgt = jnp.exp(logw - jnp.max(logw))
    # prob = wgt / jnp.sum(wgt)
    prob = lwgt_to_prob(logw)
    n_particles = logw.size
    return random.choice(key,
                         a=jnp.arange(n_particles),
                         shape=(n_particles,), p=prob)


def resample_mvn_for(key, x_particles_prev, logw):
    r"""
    Particle resampler with Multivariate Normal approximation using for-loop for testing.

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
    particle_shape = x_particles_prev.shape
    n_particles = particle_shape[0]
    prob = lwgt_to_prob(logw)
    flat = x_particles_prev.reshape((n_particles, -1))
    n_dim = flat.shape[1]
    mu = jnp.average(flat, axis=0, weights=prob)
    cov_mat = jnp.zeros((n_dim, n_dim))
    for i in range(n_dim):
        # cov_mat = cov_mat.at[i, i].set(jnp.cov(flat[:, i], aweights=prob)) # diagonal cov matrix:
        for j in range(i, n_dim):
            c = jnp.cov(flat[:, i], flat[:, j], aweights=prob)
            cov_mat = cov_mat.at[i, j].set(c[0][1])
            cov_mat = cov_mat.at[j, i].set(cov_mat[i, j])
    cov_mat += jnp.diag(jnp.ones(n_dim) * 1e-10)  # for numeric stability
    samples = random.multivariate_normal(key,
                                         mean=mu,
                                         cov=cov_mat,
                                         shape=(n_particles,))
    ret_val = {"x_particles": samples.reshape(x_particles_prev.shape),
               "mvn_mean": mu,
               "mvn_cov": cov_mat}
    return ret_val


def particle_filter_for(model, key, y_meas, theta, n_particles):
    r"""
    Apply particle filter for given value of `theta`.

    Closely follows Algorithm 2 of Murray 2013 <https://arxiv.org/abs/1306.3277>.

    This is the testing version which does the following:

    - Uses for-loops instead of `lax.scan` and `vmap/xmap`.
    - Only does basic particle sampling using `resample_multinomial_old()`.

    Args:
        model: Object specifying the state-space model.
        key: PRNG key.
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        n_particles: Number of particles.

    Returns:
        A dictionary with elements:
            - `x_particles`: An `ndarray` with leading dimensions `(n_obs, n_particles)` containing the state variable particles.
            - `logw`: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.
            - `ancestors`: An integer `ndarray` of shape `(n_obs-1, n_particles)` where each element gives the index of the particle's ancestor at the previous time point.  Since the first time point does not have ancestors, the leading dimension is `n_obs-1` instead of `n_obs`.
    """
    # memory allocation
    n_obs = y_meas.shape[0]
    # x_particles = jnp.zeros((n_obs, n_particles) + model.n_state)
    logw = jnp.zeros((n_obs, n_particles))
    ancestors = jnp.zeros((n_obs-1, n_particles), dtype=int)
    x_particles = []
    # # initial particles have no ancestors
    # ancestors = ancestors.at[0].set(-1)
    # initial time point
    key, *subkeys = random.split(key, num=n_particles+1)
    x_part = []
    for p in range(n_particles):
        xp, lw = model.pf_init(subkeys[p],
                               y_init=y_meas[0],
                               theta=theta)
        x_part.append(xp)
        # x_particles = x_particles.at[0, p].set(xp)
        logw = logw.at[0, p].set(lw)
        # x_particles = x_particles.at[0, p].set(
        #     model.init_sample(subkeys[p], y_meas[0], theta)
        # )
        # logw = logw.at[0, p].set(
        #     model.init_logw(x_particles[0, p], y_meas[0], theta)
        # )
    x_particles.append(x_part)
    # subsequent time points
    for t in range(1, n_obs):
        # resampling step
        key, subkey = random.split(key)
        ancestors = ancestors.at[t-1].set(
            resample_multinomial_old(subkey, logw[t-1])
        )
        # update
        key, *subkeys = random.split(key, num=n_particles+1)
        x_part = []
        for p in range(n_particles):
            xp, lw = model.pf_step(
                subkeys[p],
                # x_prev=x_particles[t-1, ancestors[t-1, p]],
                x_prev=x_particles[t-1][ancestors[t-1, p]],
                y_curr=y_meas[t],
                theta=theta
            )
            x_part.append(xp)
            # x_particles = x_particles.at[t, p].set(xp)
            logw = logw.at[t, p].set(lw)
            # x_particles = x_particles.at[t, p].set(
            #     model.state_sample(subkeys[p],
            #                        x_particles[t-1, ancestors[t-1, p]],
            #                        theta)
            # )
            # logw = logw.at[t, p].set(
            #     model.meas_lpdf(y_meas[t], x_particles[t, p], theta)
            # )
        x_particles.append(x_part)
    return {
        "x_particles": jnp.array(x_particles),
        "logw": logw,
        "ancestors": ancestors
    }


def particle_filter_rb_for(model, key, y_meas, theta, n_particles,
                           resampler=resample_multinomial,
                           score=True, fisher=False, history=False):
    r"""
    Rao-Blackwellized particle filter.

    This is the for-loop version used only for testing.

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

    def pf_prop(x_curr, x_prev, y_curr):
        """
        Calculate log-density of the proposal distribution `x_curr ~ q(x_t | x_t-1, y_t, theta)`.
        """
        return model.step_lpdf(x_curr=x_curr, x_prev=x_prev, y_curr=y_curr,
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
                    model.step_lpdf(
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
            prob = lwgt_to_prob(last["logw_bar"])
            alpha = last["alpha"]
            beta = last["beta"]
            alpha, gamma = jax.vmap(
                lambda p, a, b: (p * a, p * (jnp.outer(a, a) + b))
            )(prob, alpha, beta)
            alpha = jnp.sum(alpha, axis=0)
            hess = jnp.sum(gamma, axis=0) - jnp.outer(alpha, alpha)
            full["score"] = alpha
            full["fisher"] = -1. * hess

    return full


def loglik_full_for(model, y_meas, x_state, theta):
    """
    Calculate the joint loglikelihood `p(y_{0:T} | x_{0:T}, theta) * p(x_{0:T} | theta)`.

    For-loop version for testing.

    Args:
        model: Object specifying the state-space model.
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        x_state: The sequence of `n_obs` state variables `x_state = (x_0, ..., x_T)`.
        theta: Parameter value.

    Returns:
        The value of the loglikelihood.
    """
    n_obs = y_meas.shape[0]
    loglik = model.meas_lpdf(y_curr=y_meas[0], x_curr=x_state[0],
                             theta=theta)
    for t in range(1, n_obs):
        loglik = loglik + \
            model.state_lpdf(x_curr=x_state[t], x_prev=x_state[t-1],
                             theta=theta)
        loglik = loglik + \
            model.meas_lpdf(y_curr=y_meas[t], x_curr=x_state[t],
                            theta=theta)
    return loglik


def simulate_for(model, key, n_obs, x_init, theta):
    """
    Simulate data from the state-space model.

    **FIXME:** This is the testing version which uses a for-loop.  This should be put in a separate class in a `test` subfolder.

    Args:
        model: Object specifying the state-space model.
        key: PRNG key.
        n_obs: Number of observations to generate.
        x_init: Initial state value at time `t = 0`.
        theta: Parameter value.

    Returns:
        y_meas: The sequence of measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        x_state: The sequence of state variables `x_state = (x_0, ..., x_T)`, where `T = n_obs-1`.
    """
    x_state = []
    y_meas = []
    # initial observation
    key, subkey = random.split(key)
    x_state.append(x_init)
    y_meas.append(model.meas_sample(subkey, x_init, theta))
    # subsequent observations
    for t in range(1, n_obs):
        key, *subkeys = random.split(key, num=3)
        x_state.append(model.state_sample(subkeys[0], x_state[t-1], theta))
        y_meas.append(model.meas_sample(subkeys[1], x_state[t], theta))
    return jnp.array(y_meas), jnp.array(x_state)
    # y_meas = jnp.zeros((n_obs, ) + model.n_meas)
    # x_state = jnp.zeros((n_obs, ) + model.n_state)
    # x_state = x_state.at[0].set(x_init)
    # # initial observation
    # key, subkey = random.split(key)
    # y_meas = y_meas.at[0].set(model.meas_sample(subkey, x_init, theta))
    # for t in range(1, n_obs):
    #     key, *subkeys = random.split(key, num=3)
    #     x_state = x_state.at[t].set(
    #         model.state_sample(subkeys[0], x_state[t-1], theta)
    #     )
    #     y_meas = y_meas.at[t].set(
    #         model.meas_sample(subkeys[1], x_state[t], theta)
    #     )
    # return y_meas, x_state


def param_mwg_update_for(model, prior, key, theta, x_state, y_meas, rw_sd, theta_order):
    """
    Parameter update by Metropolis-within-Gibbs random walk.

    Version for testing using for-loops.

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
    n_updates = theta_order.size
    theta_curr = theta + 0.  # how else to copy...
    accept = jnp.empty(0, dtype=bool)
    lp_curr = loglik_full(model, y_meas, x_state,
                          theta_curr) + prior.lpdf(theta_curr)
    for i in theta_order:
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
        # breakpoint()
        # update parameter draw
        acc = random.bernoulli(key=subkeys[1],
                               p=jnp.minimum(1.0, jnp.exp(lrate)))
        # print("acc = {}".format(acc))
        theta_curr = theta_curr.at[i].set(
            theta_prop[i] * acc + theta_curr[i] * (1-acc)
        )
        lp_curr = lp_prop * acc + lp_curr * (1-acc)
        accept = jnp.append(accept, acc)
    return theta_curr, accept


def particle_smooth_for(key, logw, x_particles, ancestors, n_sample=1):
    r"""
    Draw a sample from `p(x_state | x_meas, theta)` using the basic particle smoothing algorithm.

    For-loop version for testing.
    """
    # wgt = jnp.exp(logw - jnp.max(logw))
    # prob = wgt / jnp.sum(wgt)
    prob = lwgt_to_prob(logw)
    n_particles = logw.size
    n_obs = x_particles.shape[0]
    n_state = x_particles.shape[2:]
    x_state = jnp.zeros((n_obs,) + n_state)
    # draw index of particle at time T
    i_part = random.choice(key, a=jnp.arange(n_particles), p=prob)
    x_state = x_state.at[n_obs-1].set(x_particles[n_obs-1, i_part, ...])
    for i_obs in reversed(range(n_obs-1)):
        # successively extract the ancestor particle going backwards in time
        i_part = ancestors[i_obs, i_part]
        x_state = x_state.at[i_obs].set(x_particles[i_obs, i_part, ...])
    return x_state


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


def particle_ancestor(x_particles, ancestors, id_particle_last):
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
    def _particle_ancestor(id_particle_next, t):
        # ancestor particle index
        id_particle_curr = ancestors[t, id_particle_next]
        return id_particle_curr, id_particle_curr

    # lax.scan
    id_particle_first, id_particle_full = \
        jax.lax.scan(_particle_ancestor, id_particle_last,
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
        lambda i: particle_ancestor(x_particles, ancestors, i)
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
        return tree_mean(acc_out, logw)
    else:
        return acc_out
