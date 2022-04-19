import unittest
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import pfjax as pf
import pfjax.mcmc as mcmc
import pfjax.models
import lotvol_model as lv
import pfjax.models.pgnet_model as pg


def rel_err(X1, X2):
    """
    Relative error between two JAX arrays.

    Adds 0.1 to the denominator to avoid nan's when its equal to zero.
    """
    x1 = X1.ravel() * 1.0
    x2 = X2.ravel() * 1.0
    return jnp.max(jnp.abs((x1 - x2)/(0.1 + x1)))


def var_sim(key, size):
    """
    Generate a variance matrix of given size.
    """
    Z = random.normal(key, (size, size))
    return jnp.matmul(Z.T, Z)

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
    prob = pf.lwgt_to_prob(logw)
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
    prob = pf.lwgt_to_prob(logw)
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
    lp_curr = pf.loglik_full(model, y_meas, x_state,
                             theta_curr) + prior.lpdf(theta_curr)
    for i in theta_order:
        # 2 subkeys for each param: rw_jump and mh_accept
        key, *subkeys = random.split(key, num=3)
        # proposal
        theta_prop = theta_curr.at[i].set(
            theta_curr[i] + rw_sd[i] * random.normal(key=subkeys[0])
        )
        # acceptance rate
        lp_prop = pf.loglik_full(model, y_meas, x_state,
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
    prob = pf.lwgt_to_prob(logw)
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


# --- now some generic external methods for constructing the tests... ----------


def bm_setup(self):
    """
    Creates input arguments to tests for BMModel.
    """
    self.key = random.PRNGKey(0)
    # parameter values
    mu = 5
    sigma = 1
    tau = .1
    self.theta = jnp.array([mu, sigma, tau])
    # data specification
    self.model_args = {"dt": .1}
    self.n_obs = 5
    self.x_init = jnp.array(0.)
    # particle filter specification
    self.n_particles = 3
    # model specification
    self.Model = pf.models.BMModel


def lv_setup(self):
    """
    Creates input arguments to tests for LotVolModel.
    """
    self.key = random.PRNGKey(0)
    # parameter values
    alpha = 1.02
    beta = 1.02
    gamma = 4.
    delta = 1.04
    sigma_H = .1
    sigma_L = .2
    tau_H = .25
    tau_L = .35
    self.theta = jnp.array([alpha, beta, gamma, delta,
                            sigma_H, sigma_L, tau_H, tau_L])
    # data specification
    dt = .09
    n_res = 3
    self.model_args = {"dt": dt, "n_res": n_res}
    self.n_obs = 7
    self.x_init = jnp.block([[jnp.zeros((n_res-1, 2))],
                             [jnp.log(jnp.array([5., 3.]))]])
    self.n_particles = 25
    self.Model = pf.models.LotVolModel
    self.Model2 = lv.LotVolModel


def pg_setup(self):
    """
    Creates input arguments to tests for LotVolModel.
    """
    self.key = random.PRNGKey(0)
    # parameter values
    theta = np.array([0.1, 0.7, 0.35, 0.2, 0.1, 0.9, 0.3, 0.1])
    tau = np.array([0.15, 0.2, 0.25, 0.3])
    self.theta = np.append(theta, tau)
    # data specification
    dt = .09
    n_res = 4
    self.model_args = {"dt": dt, "n_res": n_res}
    self.n_obs = 9
    self.x_init = jnp.block([[jnp.zeros((n_res-1, 4))],
                             [jnp.log(jnp.array([8., 8., 8., 5.]))]])
    self.n_particles = 2
    self.Model = pg.PGNETModel
    self.Model2 = pg.PGNETModel


def fact_setup(self):
    """
    Creates the variables used in the tests for factorization.
    """
    key = random.PRNGKey(0)
    self.n_lat = 3  # number of dimensions of W and X
    self.n_obs = 2  # number of dimensions of Y

    # generate random values of the matrices and vectors

    key, *subkeys = random.split(key, num=4)
    self.mu_W = random.normal(subkeys[0], (self.n_lat,))
    self.Sigma_W = var_sim(subkeys[1], self.n_lat)
    self.W = random.normal(subkeys[2], (self.n_lat,))

    key, *subkeys = random.split(key, num=4)
    self.mu_XW = random.normal(subkeys[0], (self.n_lat,))
    self.Sigma_XW = var_sim(subkeys[1], self.n_lat)
    self.X = random.normal(subkeys[2], (self.n_lat,))

    key, *subkeys = random.split(key, num=4)
    self.A = random.normal(subkeys[0], (self.n_obs, self.n_lat))
    self.Omega = var_sim(subkeys[1], self.n_obs)
    self.Y = random.normal(subkeys[2], (self.n_obs,))

    # joint distribution using single mvn
    self.mu_Y = jnp.matmul(self.A, self.mu_W + self.mu_XW)
    self.Sigma_Y = jnp.linalg.multi_dot(
        [self.A, self.Sigma_W + self.Sigma_XW, self.A.T]) + self.Omega
    AS_W = jnp.matmul(self.A, self.Sigma_W)
    AS_XW = jnp.matmul(self.A, self.Sigma_W + self.Sigma_XW)
    self.mu = jnp.block([self.mu_W, self.mu_W + self.mu_XW, self.mu_Y])
    self.Sigma = jnp.block([
        [self.Sigma_W, self.Sigma_W, AS_W.T],
        [self.Sigma_W, self.Sigma_W + self.Sigma_XW, AS_XW.T],
        [AS_W, AS_XW, self.Sigma_Y]
    ])


def test_for_sim(self):
    # un-self setUp members
    key = self.key
    theta = self.theta
    x_init = self.x_init
    model_args = self.model_args
    n_obs = self.n_obs
    n_particles = self.n_particles
    model = self.Model(**model_args)
    # simulate with for-loop
    y_meas1, x_state1 = simulate_for(
        model, key, n_obs, x_init, theta)
    # simulate without for-loop
    y_meas2, x_state2 = pf.simulate(model, key, n_obs, x_init, theta)
    self.assertAlmostEqual(rel_err(y_meas1, y_meas2), 0.0)
    self.assertAlmostEqual(rel_err(x_state1, x_state2), 0.0)


def test_for_pf(self):
    # un-self setUp members
    key = self.key
    theta = self.theta
    x_init = self.x_init
    model_args = self.model_args
    n_obs = self.n_obs
    n_particles = self.n_particles
    model = self.Model(**model_args)
    # simulate without for-loop
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    # particle filter specification
    key, subkey = random.split(key)
    # pf with for-loop
    pf_out1 = particle_filter_for(model, subkey,
                                  y_meas, theta, n_particles)
    # pf without for-loop
    pf_out2 = pf.particle_filter(
        model, subkey, y_meas, theta, n_particles)
    for k in pf_out1.keys():
        with self.subTest(k=k):
            self.assertAlmostEqual(rel_err(pf_out1[k], pf_out2[k]), 0.0)


def test_for_mvn_resampler(self):
    """ particle filter with mvn resampling function test """
    # un-self setUp members
    key = self.key
    theta = self.theta
    x_init = self.x_init
    model_args = self.model_args
    n_obs = self.n_obs
    n_particles = self.n_particles
    model = self.Model(**model_args)
    # simulate without for-loop
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    # generate initial particles:
    key, *subkeys = random.split(key, num=n_particles+1)
    x_particles, logw = jax.vmap(
        lambda k: model.pf_init(k, y_meas[0], theta))(jnp.array(subkeys))
    # x_particles = jnp.expand_dims(x_particles, 1)
    new_particles_for = resample_mvn_for(
        subkey,
        x_particles,
        logw)
    new_particles = pf.resample_mvn(
        subkey,
        x_particles,
        logw)
    for k in new_particles.keys():
        with self.subTest(k=k):
            self.assertAlmostEqual(
                rel_err(new_particles[k], new_particles_for[k]), 0.0)


def test_mvn_resample_shape(self):
    """ particle filter with mvn resampling function test """
    # un-self setUp members
    key = self.key
    key, subkey = random.split(key)
    n_particles = 25
    logw = jnp.zeros(n_particles)
    particles = jax.random.normal(subkey, shape=(n_particles, 5, 2, 2))
    new_particles_for = resample_mvn_for(
        subkey,
        particles,
        logw)
    new_particles = pf.resample_mvn(
        subkey,
        particles,
        logw)
    for k in new_particles.keys():
        with self.subTest(k=k):
            self.assertAlmostEqual(
                new_particles[k].shape, new_particles_for[k].shape)


def test_for_smooth(self):
    # un-self setUp members
    key = self.key
    theta = self.theta
    x_init = self.x_init
    model_args = self.model_args
    n_obs = self.n_obs
    n_particles = self.n_particles
    model = self.Model(**model_args)
    # simulate without for-loop
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    # pf without for-loop
    key, subkey = random.split(key)
    pf_out = pf.particle_filter(
        model, subkey, y_meas, theta, n_particles)
    # pf_smooth with for-loop
    key, subkey = random.split(key)
    x_state1 = particle_smooth_for(
        key=subkey,
        logw=pf_out["logw"][n_obs-1],
        x_particles=pf_out["x_particles"],
        ancestors=pf_out["ancestors"]
    )
    # pf_smooth without for-loop
    x_state2 = pf.particle_smooth(
        key=subkey,
        logw=pf_out["logw"][n_obs-1],
        x_particles=pf_out["x_particles"],
        ancestors=pf_out["ancestors"]
    )
    self.assertAlmostEqual(rel_err(x_state1, x_state2), 0.0)


def test_for_loglik(self):
    # un-self setUp members
    key = self.key
    theta = self.theta
    x_init = self.x_init
    model_args = self.model_args
    n_obs = self.n_obs
    n_particles = self.n_particles
    model = self.Model(**model_args)
    # simulate without for-loop
    y_meas, x_state = pf.simulate(model, key, n_obs, x_init, theta)
    # joint loglikelihood with for-loop
    loglik1 = loglik_full_for(model,
                              y_meas, x_state, theta)
    # joint loglikelihood with vmap
    loglik2 = pf.loglik_full(model,
                             y_meas, x_state, theta)
    self.assertAlmostEqual(rel_err(loglik1, loglik2), 0.0)


def test_for_mwg(self):
    # un-self setUp members
    key = self.key
    theta = self.theta
    x_init = self.x_init
    model_args = self.model_args
    n_obs = self.n_obs
    n_particles = self.n_particles
    model = self.Model(**model_args)
    # simulate without for-loop
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    # mwg setup
    prior = mcmc.NormalDiagPrior(loc=theta, scale=jnp.abs(theta))
    rw_sd = jnp.array([.1] * theta.size)
    # with default order
    theta_order = jnp.arange(theta.size)
    key, subkey = random.split(key)
    mwg_out1 = param_mwg_update_for(model, prior, subkey, theta,
                                    x_state, y_meas, rw_sd, theta_order)
    mwg_out2 = mcmc.param_mwg_update(model, prior, subkey, theta,
                                     x_state, y_meas, rw_sd, theta_order)
    for i in range(2):
        with self.subTest(i=i):
            self.assertAlmostEqual(rel_err(mwg_out1[i], mwg_out2[i]), 0.0)
    # with non-default order
    key, subkey = random.split(key)
    n_updates = 10
    theta_order = random.choice(
        subkey, jnp.arange(theta.size), shape=(n_updates,))
    key, subkey = random.split(key)
    mwg_out1 = param_mwg_update_for(model, prior, subkey, theta,
                                    x_state, y_meas, rw_sd, theta_order)
    mwg_out2 = mcmc.param_mwg_update(model, prior, subkey, theta,
                                     x_state, y_meas, rw_sd, theta_order)
    for i in range(2):
        with self.subTest(i=i):
            self.assertAlmostEqual(rel_err(mwg_out1[i], mwg_out2[i]), 0.0)


def test_jit_sim(self):
    # un-self setUp members
    key = self.key
    theta = self.theta
    x_init = self.x_init
    model_args = self.model_args
    n_obs = self.n_obs
    n_particles = self.n_particles
    model = self.Model(**model_args)
    # simulate without jit
    y_meas1, x_state1 = pf.simulate(model, key, n_obs, x_init, theta)
    # simulate with jit
    simulate_jit = jax.jit(pf.simulate, static_argnums=(0, 2))
    y_meas2, x_state2 = simulate_jit(model, key, n_obs, x_init, theta)
    self.assertAlmostEqual(rel_err(y_meas1, y_meas2), 0.0)
    self.assertAlmostEqual(rel_err(x_state1, x_state2), 0.0)
    # objective function for gradient
    def obj_fun(model, key, n_obs, x_init, theta): return jnp.mean(
        pf.simulate(model, key, n_obs, x_init, theta)[0])
    # grad without jit
    grad1 = jax.grad(obj_fun, argnums=4)(
        model, key, n_obs, x_init, theta)
    # grad with jit
    grad2 = jax.jit(jax.grad(obj_fun, argnums=4), static_argnums=(0, 2))(
        model, key, n_obs, x_init, theta)
    self.assertAlmostEqual(rel_err(grad1, grad2), 0.0)


def test_jit_pf(self):
    # un-self setUp members
    key = self.key
    theta = self.theta
    x_init = self.x_init
    model_args = self.model_args
    n_obs = self.n_obs
    n_particles = self.n_particles
    model = self.Model(**model_args)
    # simulate data
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    # particle filter specification
    key, subkey = random.split(key)
    # pf without jit
    pf_out1 = pf.particle_filter(
        model, subkey, y_meas, theta, n_particles)
    # pf with jit
    pf_out2 = jax.jit(pf.particle_filter, static_argnums=(0, 4))(
        model, subkey, y_meas, theta, n_particles)
    for k in pf_out1.keys():
        with self.subTest(k=k):
            self.assertAlmostEqual(rel_err(pf_out1[k], pf_out2[k]), 0.0)

    # objective function for gradient
    def obj_fun(model, key, y_meas, theta, n_particles):
        return pf.particle_loglik(pf.particle_filter(
            model, key, y_meas, theta, n_particles)["logw"])
    # grad without jit
    grad1 = jax.grad(obj_fun, argnums=3)(
        model, key, y_meas, theta, n_particles)
    # grad with jit
    grad2 = jax.jit(jax.grad(obj_fun, argnums=3), static_argnums=(0, 4))(
        model, key, y_meas, theta, n_particles)
    self.assertAlmostEqual(rel_err(grad1, grad2), 0.0)


def test_jit_pf_mvn(self):
    # un-self setUp members
    key = self.key
    theta = self.theta
    x_init = self.x_init
    model_args = self.model_args
    n_obs = self.n_obs
    n_particles = self.n_particles
    model = self.Model(**model_args)
    # simulate data
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    # particle filter specification
    key, subkey = random.split(key)
    # pf without jit
    pf_out1 = pf.particle_filter(
        model, subkey, y_meas, theta, n_particles,
        resampler=pf.resample_mvn)
    # pf with jit
    pf_out2 = jax.jit(pf.particle_filter, static_argnums=(0, 4, 5))(
        model, subkey, y_meas, theta, n_particles,
        resampler=pf.resample_mvn)
    for k in pf_out1.keys():
        with self.subTest(k=k):
            self.assertAlmostEqual(rel_err(pf_out1[k], pf_out2[k]), 0.0)
    # objective function for gradient

    def obj_fun(model, key, y_meas, theta, n_particles):
        return pf.particle_loglik(pf.particle_filter(
            model, key, y_meas, theta, n_particles,
            resampler=pf.resample_mvn)["logw"])

    # grad without jit
    grad1 = jax.grad(obj_fun, argnums=3)(
        model, key, y_meas, theta, n_particles)
    # grad with jit
    grad2 = jax.jit(jax.grad(obj_fun, argnums=3), static_argnums=(0, 4))(
        model, key, y_meas, theta, n_particles)
    self.assertAlmostEqual(rel_err(grad1, grad2), 0.0)


def test_jit_smooth(self):
    # un-self setUp members
    key = self.key
    theta = self.theta
    x_init = self.x_init
    model_args = self.model_args
    n_obs = self.n_obs
    n_particles = self.n_particles
    model = self.Model(**model_args)
    # simulate data
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    # particle filter specification
    key, subkey = random.split(key)
    # pf without jit
    pf_out = pf.particle_filter(
        model, subkey, y_meas, theta, n_particles)
    # pf_smooth without jit
    key, subkey = random.split(key)
    x_state1 = pf.particle_smooth(
        key=subkey,
        logw=pf_out["logw"][n_obs-1],
        x_particles=pf_out["x_particles"],
        ancestors=pf_out["ancestors"]
    )
    # pf_smooth with jit
    x_state2 = jax.jit(pf.particle_smooth)(
        key=subkey,
        logw=pf_out["logw"][n_obs-1],
        x_particles=pf_out["x_particles"],
        ancestors=pf_out["ancestors"]
    )
    self.assertAlmostEqual(rel_err(x_state1, x_state2), 0.0)

    # objective function for gradient
    def obj_fun(model, key, y_meas, theta, n_particles):
        pf_out = pf.particle_filter(model, key, y_meas, theta, n_particles)
        return jnp.sum(pf.particle_smooth(
            key=subkey,
            logw=pf_out["logw"][n_obs-1],
            x_particles=pf_out["x_particles"],
            ancestors=pf_out["ancestors"]
        ))
    # grad without jit
    grad1 = jax.grad(obj_fun, argnums=3)(
        model, key, y_meas, theta, n_particles)
    # grad with jit
    grad2 = jax.jit(jax.grad(obj_fun, argnums=3), static_argnums=(0, 4))(
        model, key, y_meas, theta, n_particles)
    self.assertAlmostEqual(rel_err(grad1, grad2), 0.0)


def test_jit_loglik(self):
    # un-self setUp members
    key = self.key
    theta = self.theta
    x_init = self.x_init
    model_args = self.model_args
    n_obs = self.n_obs
    n_particles = self.n_particles
    model = self.Model(**model_args)
    # simulate data
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    # joint loglikelihood without jit
    loglik1 = pf.loglik_full(model,
                             y_meas, x_state, theta)
    # joint loglikelihood with jit
    loglik_full_jit = jax.jit(pf.loglik_full, static_argnums=0)
    loglik2 = loglik_full_jit(model,
                              y_meas, x_state, theta)
    self.assertAlmostEqual(rel_err(loglik1, loglik2), 0.0)
    # grad without jit
    grad1 = jax.grad(pf.loglik_full, argnums=(2, 3))(
        model, y_meas, x_state, theta)
    # grad with jit
    grad2 = jax.jit(jax.grad(pf.loglik_full, argnums=(2, 3)),
                    static_argnums=0)(model, y_meas, x_state, theta)
    for i in range(2):
        with self.subTest(i=i):
            self.assertAlmostEqual(rel_err(grad1[i], grad2[i]), 0.0)


def test_jit_mwg(self):
    # un-self setUp members
    key = self.key
    theta = self.theta
    x_init = self.x_init
    model_args = self.model_args
    n_obs = self.n_obs
    n_particles = self.n_particles
    model = self.Model(**model_args)
    # simulate data
    key, subkey = random.split(key)
    y_meas, x_state = pf.simulate(model, subkey, n_obs, x_init, theta)
    # mwg setup
    prior = mcmc.NormalDiagPrior(loc=theta, scale=jnp.abs(theta))
    rw_sd = jnp.array([.1] * theta.size)
    theta_order = jnp.arange(theta.size)
    # mwg update without jit
    key, subkey = random.split(key)
    mwg_out1 = mcmc.param_mwg_update(model, prior, subkey, theta,
                                     x_state, y_meas, rw_sd, theta_order)
    # mwg update with jit
    mwg_out2 = jax.jit(mcmc.param_mwg_update,
                       static_argnums=(0, 1))(model, prior, subkey, theta,
                                              x_state, y_meas, rw_sd,
                                              theta_order)
    for i in range(2):
        with self.subTest(i=i):
            self.assertAlmostEqual(rel_err(mwg_out1[i], mwg_out2[i]), 0.0)

    # objective function for gradient
    def obj_fun(model, prior, key, theta, x_state, y_meas,
                rw_sd, theta_order):
        theta_update, accept = mcmc.param_mwg_update(
            model, prior, key, theta,
            x_state, y_meas, rw_sd, theta_order)
        return jnp.sum(theta_update)
    # grad without jit
    grad1 = jax.grad(obj_fun, argnums=(3, 4, 5))(
        model, prior, subkey, theta,
        x_state, y_meas, rw_sd, theta_order)
    # grad with jit
    grad2 = jax.jit(jax.grad(obj_fun, argnums=(3, 4, 5)),
                    static_argnums=(0, 1))(model, prior, subkey, theta,
                                           x_state, y_meas, rw_sd,
                                           theta_order)
    for i in range(3):
        with self.subTest(i=i):
            self.assertAlmostEqual(rel_err(grad1[i], grad2[i]), 0.0)


def test_models_sim(self):
    # un-self setUp members
    key = self.key
    theta = self.theta
    x_init = self.x_init
    model_args = self.model_args
    n_obs = self.n_obs
    n_particles = self.n_particles
    model1 = self.Model(**model_args)
    model2 = self.Model2(**model_args)
    # simulate with non-inherited class
    y_meas1, x_state1 = pf.simulate(model1, key, n_obs, x_init, theta)
    # simulate with inherited class
    y_meas2, x_state2 = pf.simulate(model2, key, n_obs, x_init, theta)
    self.assertAlmostEqual(rel_err(y_meas1, y_meas2), 0.0)
    self.assertAlmostEqual(rel_err(x_state1, x_state2), 0.0)


def test_models_loglik(self):
    # un-self setUp members
    key = self.key
    theta = self.theta
    x_init = self.x_init
    model_args = self.model_args
    n_obs = self.n_obs
    n_particles = self.n_particles
    model1 = self.Model(**model_args)
    model2 = self.Model2(**model_args)
    # simulate with inherited class
    y_meas, x_state = pf.simulate(model2, key, n_obs, x_init, theta)
    # joint loglikelihood with non-inherited class
    loglik1 = pf.loglik_full(model1,
                             y_meas, x_state, theta)
    # joint loglikelihood with inherited class
    loglik2 = pf.loglik_full(model2,
                             y_meas, x_state, theta)
    self.assertAlmostEqual(rel_err(loglik1, loglik2), 0.0)


def test_models_pf(self):
    # un-self setUp members
    key = self.key
    theta = self.theta
    x_init = self.x_init
    model_args = self.model_args
    n_obs = self.n_obs
    n_particles = self.n_particles
    model1 = self.Model(**model_args)
    model2 = self.Model2(**model_args)
    # simulate with inherited class
    y_meas, x_state = pf.simulate(model2, key, n_obs, x_init, theta)
    # particle filter specification
    key, subkey = random.split(key)
    # pf with non-inherited class
    pf_out1 = pf.particle_filter(
        model1, subkey, y_meas, theta, n_particles)
    # pf with inherited class
    pf_out2 = pf.particle_filter(
        model2, subkey, y_meas, theta, n_particles)
    for k in pf_out1.keys():
        with self.subTest(k=k):
            self.assertAlmostEqual(rel_err(pf_out1[k], pf_out2[k]), 0.0)
