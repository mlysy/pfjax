"""
Particle filter using multivariate normal distribution in JAX.

Uses the same API as NumPy/SciPy version.

### Variant 1: 
- Removing the X_particles matrix because it makes no sense for MVN, also getting some issues with assignment of samples from MVN to matrix.
    Instead, going to leave X_particles as a single (n_particles, n_states) martix that gets updated every iteration 
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import lax
from jax.experimental.maps import xmap
from functools import partial


@jax.jit
def _lweight_to_prob(logw):
    """Returns normalized propabilities from unnormalized log weights

    Args:
        logw ([type]): [description]
    """
    wgt = jnp.exp(logw - jnp.max(logw))  # jnp.maximum
    prob = wgt / jnp.sum(wgt)
    return prob

# @jax.jit
def particle_resample_mvn(particles, logw, key):
    """
    Approximate particle distribution with MVN. Uses weighted mean and covariance of `particles` for the MVN mean and cov

    Args:
        particles: Matrix of size (n_particles, n_states) containing particles at the current timestep
        logw: Vector of `n_particles` unnormalized log-weights
        key: PRNG key

    Returns: 
        Matrix of size (`n_particles`, `n_states`) representing the particles for the next timestep 
    """
    assert len(particles.shape) == 2, "Invalid particle shape, must have dimension (n_particles, n_states), got {0}".format(particles.shape)
    n_particles, n_states = particles.shape
    prob = _lweight_to_prob(logw)

    # variables are the rows in jnp.cov()
    cov_mat = jnp.cov(jnp.transpose(particles), aweights=prob)
    mu = jnp.average(particles, axis=0, weights=prob).reshape(-1, )

    if n_states == 1:
        cov_mat = cov_mat.reshape(-1, 1) # change int into matrix

    samples = random.multivariate_normal(key, mean=mu,
                                         cov=cov_mat,
                                         shape=(n_particles, n_states))
    return jnp.squeeze(samples), mu, cov_mat
    

def particle_filter_for(model, y_meas, theta, n_particles, key):
    """
    Apply particle filter for given value of `theta`.

    Closely follows Algorithm 2 of https://arxiv.org/pdf/1306.3277.pdf.

    This is the testing version which does the following:

    - Uses for-loops instead of `lax.scan` and `vmap/xmap`.
    - Only performs a bootstrap particle filter using `state_sample()` and `meas_lpdf()`.

    **FIXME:** Refactor `ancestor_particles` to use particle_resample_mvn

    Args:
        model: Object specifying the state-space model.
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        n_particles: Number of particles.
        key: PRNG key.

    Returns:
        A dictionary with elements:
            - `X_particles_mu`: An `ndarray` with leading dimensions `(n_obs, n_particles)` containing the state variable particles.
            - `logw_particles`: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.
    """
    # memory allocation
    n_obs = y_meas.shape[0]
    X_particles_mu = jnp.zeros((n_obs, model.n_meas))
    X_particles = jnp.zeros((n_particles, model.n_meas))
    logw_particles = jnp.zeros((n_obs, n_particles))
    ancestor_particles = jnp.zeros((n_obs, n_particles), dtype=int)
    # initial particles have no ancestors
    ancestor_particles = ancestor_particles.at[0].set(-1)
    # initial time point
    key, *subkeys = random.split(key, num=n_particles+1)
    for p in range(n_particles):
        X_particles = X_particles.at[p].set(
            model.init_sample(y_meas[0], theta, subkeys[p])
        )
        logw_particles = logw_particles.at[0, p].set(
            model.init_logw(X_particles[p], y_meas[0], theta)
        )
    # subsequent time points
    for t in range(1, n_obs):
        # resampling step
        key, subkey = random.split(key)
        resampled_particles = particle_resample_mvn(
            jnp.squeeze(X_particles), logw_particles[t-1], subkey)
        X_particles = resampled_particles[0]
        X_particles_mu = X_particles_mu.at[t].set(resampled_particles[1]) # mean of MVN - UNCOMMENT

        # update
        key, *subkeys = random.split(key, num=n_particles+1)
        for p in range(n_particles):
            X_particles = X_particles.at[p].set(model.state_sample(
                X_particles[p], theta, subkeys[p])
            )
            logw_particles = logw_particles.at[t, p].set(
                model.meas_lpdf(y_meas[t], X_particles[p], theta)
            )

    return {
        "X_particles_mu": X_particles_mu,  
        "logw_particles": logw_particles,
    }


# @partial(jax.jit, static_argnums=2)
def particle_filter(model, y_meas, theta, n_particles, key):
    """
    Apply particle filter using a MVN approximation of the particle distribution for given value of `theta`.
    This model does not return `ancestor_particles`, as there is no ancestry for the MVN

    Closely follows Algorithm 2 of https://arxiv.org/pdf/1306.3277.pdf.

    Args:
        model: Object specifying the state-space model.
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        n_particles: Number of particles.
        key: PRNG key.

    Returns:
        A dictionary with elements:
            - `X_particles_mu`: An `ndarray` with leading dimensions `(n_obs, n_particles)` containing the mean of the MVN at each timestep. Note that this is different from the vanilla particle filter, which returns the particles at each timestep
            - `logw_particles`: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.
    """
    n_obs = y_meas.shape[0]

    # lax.scan setup
    # scan function
    def fun(carry, t):
        """ 
        Simple way: Sample particles and use that as the carry. This defeats the purpose of using the MVN because we still carry around the particles. Should update this later
        """
        # resampling step
        key, subkey = random.split(carry["key"])
        resampled_particles = particle_resample_mvn(jnp.squeeze(carry["X_particles"]), 
                                                    carry["logw_particles"], 
                                                    subkey)
        X_particles = resampled_particles[0]
        X_particles_mu = resampled_particles[1]

        # update particles
        key, *subkeys = random.split(key, num=n_particles+1)
        X_particles, logw_particles = jax.vmap(
            lambda xs, k: model.pf_step(xs, y_meas[t], theta, k)
        )(X_particles, jnp.array(subkeys))

        res = {
            "logw_particles": logw_particles,
            "X_particles": X_particles,
            "X_particles_mu": X_particles_mu,
            "key": key
        }
        return res, res
    # scan initial value
    key, *subkeys = random.split(key, num=n_particles+1)
    # vmap version
    X_particles_init, logw_particles_init = jax.vmap(
        lambda k: model.pf_init(y_meas[0], theta, k))(jnp.array(subkeys))
    prob = _lweight_to_prob(logw_particles_init)
    init = {
        "X_particles": X_particles_init,
        "logw_particles": logw_particles_init,
        "X_particles_mu": jnp.average(X_particles_init, axis=0, weights=prob).reshape(-1, ),
        "key": key
    }

    # lax.scan itself
    last, full = lax.scan(fun, init, jnp.arange(1, n_obs))
    # append initial values
    out = {
        k: jnp.append(jnp.expand_dims(init[k], axis=0), full[k], axis=0)
        for k in ["logw_particles", "X_particles_mu"]
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
    return jnp.sum(jsp.special.logsumexp(logw_particles, axis=1))


def get_sum_lweights_mvn(theta, key, n_particles, y_meas, model):
    """
    FIXME: Remove this function, should only be one `get_sum_lweights` for all particle filters
    Args:
        theta: A `jnp.array` that represents the values of the parameters.
        key: The key required for the prng.
        n_particles: The number of particles to use in the particle filter.
        y_meas: The measurements of the observations required for the particle filter.

    Returns:
        The sum of the particle log weights from the particle filters.
    """

    ret = particle_filter(model, y_meas, theta, n_particles, key)
    sum_particle_lweights = particle_loglik(ret['logw_particles'])
    return sum_particle_lweights


def density_estimator (particles, weights, key):
    """
    General signiture for the density estimation function in particle filters. 

    Args:
        particles: particles 
        weights: unnormalized log-weights corresponding to each particles. Must be same length as `particles`
        key: jax key

    Returns:
        samples: len(particles) number of samples from the estimated density 
        x_particles_summ: summary statistic for particles. Could be the mean if density estimation uses a MVN, etc. 
    """
    pass


def log_posterior(theta, key, n_particles, y_meas, model, prior):
    """log posterior. 

    Args:
        params ([type]): jnp array of MLE parameter estiamtes calculated by `stoch_opt` or another optim function 
        loglik ([type]): log-likelihood
    """
    ret = particle_filter(model, y_meas, theta, n_particles, key)
    sum_particle_lweights = particle_loglik(ret['logw_particles'])
    lprior = jnp.log(prior(theta))
    return sum_particle_lweights + lprior
