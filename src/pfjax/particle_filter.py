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
from functools import partial


def particle_resample(logw, key):
    """
    Particle resampler.

    This basic one just does a multinomial sampler, i.e., sample with replacement proportional to weights.

    Args:
        logw: Vector of `n_particles` unnormalized log-weights.
        key: PRNG key.

    Returns:
        Vector of `n_particles` integers between 0 and `n_particles-1`, sampled with replacement with probability vector `exp(logw) / sum(exp(logw))`.
    """
    wgt = jnp.exp(logw - jnp.max(logw))
    prob = wgt / jnp.sum(wgt)
    n_particles = logw.size
    return random.choice(key,
                         a=jnp.arange(n_particles),
                         shape=(n_particles,), p=prob)


def particle_filter_for(model, y_meas, theta, n_particles, key):
    """
    Apply particle filter for given value of `theta`.

    Closely follows Algorithm 2 of https://arxiv.org/pdf/1306.3277.pdf.

    This is the testing version which does the following:

    - Uses for-loops instead of `lax.scan` and `vmap/xmap`.
    - Only performs a bootstrap particle filter using `state_sample()` and `meas_lpdf()`.

    **FIXME:** Move this to the `tests` module.

    Args:
        model: Object specifying the state-space model.
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        n_particles: Number of particles.
        key: PRNG key.

    Returns:
        A dictionary with elements:
            - `X_particles`: An `ndarray` with leading dimensions `(n_obs, n_particles)` containing the state variable particles.
            - `logw_particles`: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.
            - `ancestor_particles`: An integer `ndarray` of shape `(n_obs, n_particles)` where each element gives the index of the particle's ancestor at the previous time point.  Since the first time point does not have ancestors, the first row of `ancestor_particles` contains all `-1`.
    """
    # memory allocation
    n_obs = y_meas.shape[0]
    X_particles = jnp.zeros((n_obs, n_particles, model.n_state))
    logw_particles = jnp.zeros((n_obs, n_particles))
    ancestor_particles = jnp.zeros((n_obs, n_particles), dtype=int)
    # initial particles have no ancestors
    ancestor_particles = ancestor_particles.at[0].set(-1)
    # initial time point
    key, *subkeys = random.split(key, num=n_particles+1)
    for p in range(n_particles):
        X_particles = X_particles.at[0, p].set(
            model.init_sample(y_meas[0], theta, subkeys[p])
        )
        logw_particles = logw_particles.at[0, p].set(
            model.init_logw(X_particles[0, p], y_meas[0], theta)
        )
    # X_particles = X_particles.at[0].set(
    #     jax.vmap(lambda k: model.init_sample(y_meas[0], theta, k))(
    #         jnp.array(subkeys)
    #     )
    # )
    # logw_particles = logw_particles.at[0].set(
    #     jax.vmap(lambda xs: model.init_logw(xs, y_meas[0], theta) +
    #              model.meas_lpdf(y_meas[0], xs, theta))(X_particles[0])
    # )
    # subsequent time points
    for t in range(1, n_obs):
        # resampling step
        key, subkey = random.split(key)
        ancestor_particles = ancestor_particles.at[t].set(
            particle_resample(logw_particles[t-1], subkey)
        )
        # update
        key, *subkeys = random.split(key, num=n_particles+1)
        for p in range(n_particles):
            X_particles = X_particles.at[t, p].set(
                model.state_sample(X_particles[t-1, ancestor_particles[t, p]],
                                   theta, subkeys[p])
            )
            logw_particles = logw_particles.at[t, p].set(
                model.meas_lpdf(y_meas[t], X_particles[t, p], theta)
            )
        # X_particles = X_particles.at[t].set(
        #     jax.vmap(lambda xs, k: model.state_sample(xs, theta, k))(
        #         X_particles[t-1, ancestor_particles[t]], jnp.array(subkeys)
        #     )
        # )
        # logw_particles = logw_particles.at[t].set(
        #     jax.vmap(lambda xs: model.meas_lpdf(y_meas[t], xs, theta))(
        #         X_particles[t]
        #     )
        # )
    return {
        "X_particles": X_particles,
        "logw_particles": logw_particles,
        "ancestor_particles": ancestor_particles
    }


# @partial(jax.jit, static_argnums=2)
def particle_filter(model, y_meas, theta, n_particles, key):
    """
    Apply particle filter for given value of `theta`.

    Closely follows Algorithm 2 of https://arxiv.org/pdf/1306.3277.pdf.

    Args:
        model: Object specifying the state-space model.
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        n_particles: Number of particles.
        key: PRNG key.

    Returns:
        A dictionary with elements:
            - `X_particles`: An `ndarray` with leading dimensions `(n_obs, n_particles)` containing the state variable particles.
            - `logw_particles`: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.
            - `ancestor_particles`: An integer `ndarray` of shape `(n_obs, n_particles)` where each element gives the index of the particle's ancestor at the previous time point.  Since the first time point does not have ancestors, the first row of `ancestor_particles` contains all `-1`.
    """
    n_obs = y_meas.shape[0]

    # lax.scan setup
    # scan function
    def fun(carry, t):
        # resampling step
        key, subkey = random.split(carry["key"])
        ancestor_particles = particle_resample(carry["logw_particles"],
                                               subkey)
        # update particles
        key, *subkeys = random.split(key, num=n_particles+1)
        X_particles, logw_particles = jax.vmap(
            lambda xs, k: model.pf_step(xs, y_meas[t], theta, k)
        )(carry["X_particles"][ancestor_particles], jnp.array(subkeys))
        # X_particles = jax.vmap(lambda xs, k: model.state_sample(xs, theta, k))(
        #     carry["X_particles"][ancestor_particles], jnp.array(subkeys)
        # )
        # # update log-weights
        # logw_particles = jax.vmap(
        #     lambda xs: model.meas_lpdf(y_meas[t], xs, theta)
        # )(X_particles)
        # breakpoint()
        # output
        res = {
            "ancestor_particles": ancestor_particles,
            "logw_particles": logw_particles,
            "X_particles": X_particles,
            "key": key
        }
        return res, res
    # scan initial value
    key, *subkeys = random.split(key, num=n_particles+1)
    # vmap version
    X_particles, logw_particles = jax.vmap(
        lambda k: model.pf_init(y_meas[0], theta, k))(jnp.array(subkeys))
    # X_particles = jax.vmap(
    #     lambda k: model.init_sample(y_meas[0], theta, k))(jnp.array(subkeys))
    # logw_particles = jax.vmap(
    #     lambda xs: model.init_logw(xs, y_meas[0], theta))(X_particles)
    # xmap version: experimental!
    # X_particles = xmap(
    #     lambda ym, th, k: model.init_sample(ym, th, k),
    #     in_axes=([...], [...], ["particles", ...]),
    #     out_axes=["particles", ...])(y_meas[0], theta, jnp.array(subkeys))
    # logw_particles = xmap(
    #     lambda xs, ym, th: model.init_logw(xs, ym, th),
    #     in_axes=(["particles", ...], [...], [...]),
    #     out_axes=["particles", ...])(X_particles, y_meas[0], theta)
    init = {
        "X_particles": X_particles,
        "logw_particles": logw_particles,
        "ancestor_particles": -jnp.ones(n_particles, dtype=int),
        "key": key
    }
    # lax.scan itself
    last, full = lax.scan(fun, init, jnp.arange(1, n_obs))
    # append initial values
    out = {
        k: jnp.append(jnp.expand_dims(init[k], axis=0), full[k], axis=0)
        for k in ["X_particles", "logw_particles", "ancestor_particles"]
    }
    return out


def particle_loglik(logw_particles):
    """
    Calculate particle filter marginal loglikelihood.

    FIXME: Libbi paper does `logmeanexp` instead of `logsumexp`...

    Args:
        logw_particles: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.

    Returns:
        Particle filter approximation of
        ```
        log p(y_meas | theta) = log int p(y_meas | x_state, theta) * p(x_state | theta) dx_state
        ```
    """
    n_particles = logw_particles.shape[1]
    return jnp.sum(jsp.special.logsumexp(logw_particles, axis=1) - jnp.log(n_particles))


def get_sum_lweights(theta, key, n_particles, y_meas, model):
    """

    Args:
        theta: A `jnp.array` that represents the values of the parameters.
        key: The key required for the prng.
        n_particles: The number of particles to use in the particle filter.
        y_meas: The measurements of the observations required for the particle filter.

    Returns:
        The sum of the particle log weights from the particle filters.
    """
<<<<<<< HEAD

    ret = particle_filter(model, y_meas, theta, n_particles, key)
=======
    
<<<<<<< HEAD
    ret = particle_filter(y_meas, theta, n_particles, key)
>>>>>>> adding stoch_opt code
=======
    ret = particle_filter(model, y_meas, theta, n_particles, key)
>>>>>>> refactoring stoch opt code to fit model desc + fixing implementation
    sum_particle_lweights = particle_loglik(ret['logw_particles'])
    return sum_particle_lweights


<<<<<<< HEAD
def joint_loglik_for(model, y_meas, x_state, theta):
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


def joint_loglik(model, y_meas, x_state, theta):
    """
    Calculate the joint loglikelihood `p(y_{0:T} | x_{0:T}, theta) * p(x_{0:T} | theta)`.

    Args:
        model: Object specifying the state-space model.
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        x_state: The sequence of `n_obs` state variables `x_state = (x_0, ..., x_T)`.
        theta: Parameter value.

    Returns:
        The value of the loglikelihood.
    """
    n_obs = y_meas.shape[0]
    # initial measurement
    ll_init = model.meas_lpdf(y_curr=y_meas[0], x_curr=x_state[0],
                              theta=theta)
    # subsequent measurements and state variables
    ll_step = jax.vmap(lambda t:
                       model.state_lpdf(x_curr=x_state[t],
                                        x_prev=x_state[t-1],
                                        theta=theta) +
                       model.meas_lpdf(y_curr=y_meas[t],
                                       x_curr=x_state[t],
                                       theta=theta))(jnp.arange(1, n_obs))
    return ll_init + jnp.sum(ll_step)


def update_params(params, subkey, grad_fun=None, n_particles=100, y_meas=None, model=None, learning_rate=0.01, mask=None):
    params_update = jax.grad(grad_fun, argnums=0)(
        params, subkey, n_particles, y_meas, model)
    return params + learning_rate * (jnp.where(mask, params_update, 0))


def stoch_opt(model, params, grad_fun, y_meas, n_particles=100, iterations=10,
              learning_rate=0.01, key=1, mask=None):
=======
def stoch_opt(model, params, grad_fun, y_meas, n_particles=100, iterations=10, learning_rate=0.01, key=1):
>>>>>>> refactoring stoch opt code to fit model desc + fixing implementation
    """
    Args:
    
    Args:
        params: A `jnp.array` that represents the initial values of the parameters.
        grad_fun: The function which we would like to take the gradient with respect to.
        y_meas: The measurements of the observations required for the particle filter.
        n_particles: The number of particles to use in the particle filter.
        learning_rate: The learning rate for the gradient descent algorithm.
        iterations: The number of iterations to run the gradient descent for.
        key: The key required for the prng.

    Returns:
        The stochastic approximation of theta which are the parameters of the model.
    """
<<<<<<< HEAD
<<<<<<< HEAD
    partial_update_params = partial(update_params, n_particles=n_particles, y_meas=y_meas,
                                    model=model, learning_rate=learning_rate, mask=mask, grad_fun=grad_fun)
    update_fn = jax.jit(partial_update_params, donate_argnums=(0,))
    keys = random.split(key, iterations)
    for subkey in keys:
        params = update_fn(params, subkey)
        print(params)
    return params
=======
    grad_lweights = jax.jit(jax.grad(grad_fun, key, n_particles, y_meas))
=======
    grad_lweights = jax.jit(jax.grad(grad_fun, argnums=0), static_argnums=(2,4))
>>>>>>> refactoring stoch opt code to fit model desc + fixing implementation
    for i in range(iterations):
        key, subkey = random.split(key)
        params_update = grad_lweights(params, subkey, n_particles, y_meas, model)
        params = params + (learning_rate * params_update)
        
    return params
>>>>>>> adding stoch_opt code
