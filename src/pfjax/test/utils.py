"""
Utilities for both formal and interactive testing. 
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import jax.tree_util as jtu
from ..utils import *
from ..loglik_full import loglik_full

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
