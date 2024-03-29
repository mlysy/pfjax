"""
Prototype for particle filter using NumPy/SciPy.

The API requires the user to define the following functions:

- `state_lpdf(x_curr, x_last, theta)`: Log-density of `p(x_t | x_{t-1}, theta)`.
- `state_sample(x_last, theta)`: Sample from `x_curr ~ p(x_t | x_{t-1}, theta)`.
- `meas_lpdf(y_curr, x_curr, theta)`: Log-density of `p(y_t | x_t, theta)`.
- `meas_sample(x_curr, theta)`: Sample from `y_curr ~ p(y_t | x_t, theta)`.
- `init_sample(y_init, theta)`: Sample from *proposal* distribution `x_init ~ q(x_init | y_init, theta)`.  See below for details.
- `init_logw(x_init, y_init, theta)`: Log-weights for importance sampler for `x_init`.

For now, additional inputs are specified as global constants.

Importance sampling is used to draw from the initial state variable `x_init` at time `t = 0`.  That is, the proposal distribution is
```
x_init ~ q(x_init) = q(x_init | y_init, theta)
```
as obtained by `init_sample()`.  The samples are then reweighted with log-weights
```
logw = log p(x_init | theta) - log q(x_init)
```
as calculated by `init_logw()`.

The provided functions are:

- `meas_sim(n_obs, x_init, theta)`: Obtain a sample from `y_meas = (y_1, ..., y_T)` and `x_state = (x_1, ..., x_T)`.
- `particle_filter(y_meas, theta, n_particles): Run the particle filter.
- `particle_loglik(logw_particles)`: Compute the particle filter marginal loglikelihoood.
- `particle_smooth(logw, X_particles, ancestor_particles, n_sample)`: Posterior sampling from the particle filter distribution of `p(x_state | y_meas, theta)`.
- `particle_resample(logw)`: A rudimentary particle resampling method.
"""

import numpy as np
import scipy as sp
import scipy.stats


def meas_sim(n_obs, x_init, theta):
    """
    Simulate data from the state-space model.

    Args:
        n_obs: Number of observations to generate.
        x_init: Initial state value at time `t = 0`.
        theta: Parameter value.

    Returns:
        y_meas: The sequence of measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        x_state: The sequence of state variables `x_state = (x_0, ..., x_T)`, where `T = n_obs-1` and `x_0 = x_init`.
    """
    y_meas = np.zeros((n_obs, n_meas))
    x_state = np.zeros((n_obs, n_state))
    x_state[0] = x_init
    for t in range(1, n_obs):
        x_state[t] = state_sample(x_state[t-1], theta)
        y_meas[t] = meas_sample(x_state[t], theta)
    return y_meas, x_state


def particle_resample(logw):
    """
    Particle resampler.

    This basic one just does a multinomial sampler, i.e., sample with replacement proportional to weights.

    Args:
        logw: Vector of `n_particles` unnormalized log-weights.

    Returns:
        Vector of `n_particles` integers between 0 and `n_particles-1`, sampled with replacement with probability vector `exp(logw) / sum(exp(logw))`.
    """
    wgt = np.exp(logw - np.max(logw))
    prob = wgt / np.sum(wgt)
    n_particles = logw.size
    return np.random.choice(np.arange(n_particles), size=n_particles, p=prob)


def particle_filter(y_meas, theta, n_particles):
    """
    Apply particle filter for given value of `theta`.

    Closely follows Algorithm 2 of https://arxiv.org/pdf/1306.3277.pdf.

    Args:
        y_meas: The sequence of `n_obs` measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        theta: Parameter value.
        n_particles: Number of particles.

    Returns:
        A dictionary with elements:
            - `X_particles`: An `ndarray` with leading dimensions `(n_obs, n_particles)` containing the state variable particles.
            - `logw_particles`: An `ndarray` of shape `(n_obs, n_particles)` giving the unnormalized log-weights of each particle at each time point.
            - `ancestor_particles`: An integer `ndarray` of shape `(n_obs, n_particles)` where each element gives the index of the particle's ancestor at the previous time point.  Since the first time point does not have ancestors, the first row of `ancestor_particles` contains all `-1`.
    """
    # memory allocation
    n_obs = y_meas.shape[0]
    X_particles = np.zeros((n_obs, n_particles, n_state))
    logw_particles = np.zeros((n_obs, n_particles))
    ancestor_particles = np.zeros((n_obs, n_particles), dtype=int)
    ancestor_particles[0] = -1  # initial particles have no ancestors
    # initial time point
    for i_part in range(n_particles):
        X_particles[0, i_part] = init_sample(y_meas[0], theta)
        logw_particles[0, i_part] = \
            init_logw(X_particles[0, i_part], y_meas[0], theta) + \
            meas_lpdf(y_meas[0], X_particles[0, i_part], theta)
    # subsequent time points
    for t in range(1, n_obs):
        # resampling step
        ancestor_particles[t] = particle_resample(logw_particles[t-1])
        for i_part in range(n_particles):
            X_particles[t, i_part, :] = state_sample(
                X_particles[t-1, ancestor_particles[t, i_part], :], theta
            )
            logw_particles[t, i_part] = meas_lpdf(
                y_meas[t, :], X_particles[t, i_part, :], theta
            )
    return {
        "X_particles": X_particles,
        "logw_particles": logw_particles,
        "ancestor_particles": ancestor_particles
    }


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
    return np.sum(sp.special.logsumexp(logw_particles, axis=1))


def particle_smooth(logw, X_particles, ancestor_particles, n_sample=1):
    """
    Basic particle smoothing algorithm.

    Samples from posterior distribution `p(x_state | x_meas, theta)`.

    Args:
        logw: Vector of `n_particles` unnormalized log-weights at the last time point `t = n_obs-1`.
        X_particles: An `ndarray` with leading dimensions `(n_obs, n_particles)` containing the state variable particles.        
        ancestor_particles: An integer `ndarray` of shape `(n_obs, n_particles)` where each element gives the index of the particle's ancestor at the previous time point.
        n_sample: Number of draws of `x_state` to return.

    Returns:
        An `ndarray` with leading dimension `n_sample` corresponding to as many samples from the particle filter approximation to the posterior distribution `p(x_state | x_meas, theta)`.
    """
    wgt = np.exp(logw - np.max(logw))
    prob = wgt / np.sum(wgt)
    n_particles = logw.size
    n_obs = X_particles.shape[0]
    n_state = X_particles.shape[2]
    x_state = np.zeros((n_sample, n_obs, n_state))
    for i_samp in range(n_sample):
        i_part = np.random.choice(np.arange(n_particles), size=1, p=prob)
        # i_part_T = i_part
        x_state[i_samp, n_obs-1] = X_particles[n_obs-1, i_part, :]
        for i_obs in reversed(range(n_obs-1)):
            i_part = ancestor_particles[i_obs+1, i_part]
            x_state[i_samp, i_obs] = X_particles[i_obs, i_part, :]
    return x_state  # , i_part_T
