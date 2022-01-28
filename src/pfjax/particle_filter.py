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
from jax import random
from jax import lax
from jax.experimental.maps import xmap


def particle_resample_old(key, logw):
    """
    Particle resampler.

    This basic one just does a multinomial sampler, i.e., sample with replacement proportional to weights.

    Old API, to be depreciated after testing against `particle_filter_for()`.

    Args:
        key: PRNG key.
        logw: Vector of `n_particles` unnormalized log-weights.

    Returns:
        Vector of `n_particles` integers between 0 and `n_particles-1`, sampled with replacement with probability vector `exp(logw) / sum(exp(logw))`.
    """
    wgt = jnp.exp(logw - jnp.max(logw))
    prob = wgt / jnp.sum(wgt)
    n_particles = logw.size
    return random.choice(key,
                         a=jnp.arange(n_particles),
                         shape=(n_particles,), p=prob)


def particle_resample(key, x_particles_prev, logw):
    """
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
    wgt = jnp.exp(logw - jnp.max(logw))
    prob = wgt / jnp.sum(wgt)
    n_particles = logw.size
    ancestors = random.choice(key,
                              a=jnp.arange(n_particles),
                              shape=(n_particles,), p=prob)
    return {
        "x_particles": x_particles_prev[ancestors, ...],
        "ancestors": ancestors
    }


def particle_filter_for(model, key, y_meas, theta, n_particles):
    """
    Apply particle filter for given value of `theta`.

    This is the testing version which does the following:

    - Uses for-loops instead of `lax.scan` and `vmap/xmap`.
    - Only does basic particle sampling using `particle_resample_old()`.
    """
    # memory allocation
    n_obs = y_meas.shape[0]
    x_particles = jnp.zeros((n_obs, n_particles) + model.n_state)
    logw = jnp.zeros((n_obs, n_particles))
    ancestors = jnp.zeros((n_obs-1, n_particles), dtype=int)
    # # initial particles have no ancestors
    # ancestors = ancestors.at[0].set(-1)
    # initial time point
    key, *subkeys = random.split(key, num=n_particles+1)
    for p in range(n_particles):
        xp, lw = model.pf_init(subkeys[p],
                               y_init=y_meas[0],
                               theta=theta)
        x_particles = x_particles.at[0, p].set(xp)
        logw = logw.at[0, p].set(lw)
        # x_particles = x_particles.at[0, p].set(
        #     model.init_sample(subkeys[p], y_meas[0], theta)
        # )
        # logw = logw.at[0, p].set(
        #     model.init_logw(x_particles[0, p], y_meas[0], theta)
        # )
    # subsequent time points
    for t in range(1, n_obs):
        # resampling step
        key, subkey = random.split(key)
        ancestors = ancestors.at[t-1].set(
            particle_resample_old(subkey, logw[t-1])
        )
        # update
        key, *subkeys = random.split(key, num=n_particles+1)
        for p in range(n_particles):
            xp, lw = model.pf_step(
                subkeys[p],
                x_prev=x_particles[t-1, ancestors[t-1, p]],
                y_curr=y_meas[t],
                theta=theta
            )
            x_particles = x_particles.at[t, p].set(xp)
            logw = logw.at[t, p].set(lw)
            # x_particles = x_particles.at[t, p].set(
            #     model.state_sample(subkeys[p],
            #                        x_particles[t-1, ancestors[t-1, p]],
            #                        theta)
            # )
            # logw = logw.at[t, p].set(
            #     model.meas_lpdf(y_meas[t], x_particles[t, p], theta)
            # )
    return {
        "x_particles": x_particles,
        "logw": logw,
        "ancestors": ancestors
    }


def particle_filter(model, key, y_meas, theta, n_particles,
                    particle_sampler=particle_resample):
    """
    Apply particle filter for given value of `theta`.

    Closely follows Algorithm 2 of https://arxiv.org/pdf/1306.3277.pdf.

    Args:
        model: Object specifying the state-space model.
        key: PRNG key.
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        n_particles: Number of particles.
        particle_sampler: Function used at step `t` to obtain sample of particles from `p(x_{t-1} | y_{0:t-1}, theta)`.  The inputs to the function are `particle_sampler(x_particles, logw, key)`, and the return value is a dictionary with mandatory element `x_particles` and optional elements that get stacked to the final output using `lax.scan()`.  Default value is `particle_resample()`.

    Returns:
        A dictionary with elements:
            - `x_particles`: An `ndarray` with leading dimensions `(n_obs, n_particles)` containing the state variable particles.
            - `logw`: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.
            - `...`: Other `ndarray`s with leading dimension `n_obs-1`, corresponding to additional outputs from `particle_sampler()` as accumulated by `lax.scan()`.  Since these additional outputs do not apply to the first time step (since it has no previous time step), the leading dimension of each additional output is `n_obs-1`.
    """
    n_obs = y_meas.shape[0]

    # lax.scan setup
    # scan function
    def fun(carry, t):
        # sample particles from previous time point
        key, subkey = random.split(carry["key"])
        new_particles = particle_sampler(subkey,
                                         carry["x_particles"],
                                         carry["logw"])
        # update particles to current time point (and get weights)
        key, *subkeys = random.split(key, num=n_particles+1)
        x_particles, logw = jax.vmap(
            lambda xs, k: model.pf_step(k, xs, y_meas[t], theta)
        )(new_particles["x_particles"], jnp.array(subkeys))
        # breakpoint()
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
    """
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


def particle_smooth_for(key, logw, x_particles, ancestors):
    """
    Draw a sample from `p(x_state | x_meas, theta)` using the basic particle smoothing algorithm.

    For-loop version for testing.
    """
    wgt = jnp.exp(logw - jnp.max(logw))
    prob = wgt / jnp.sum(wgt)
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


def particle_smooth(key, logw, x_particles, ancestors):
    """
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
    wgt = jnp.exp(logw - jnp.max(logw))
    prob = wgt / jnp.sum(wgt)

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
