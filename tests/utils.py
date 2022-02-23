import unittest
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import pfjax as pf
import pfjax.mcmc as mcmc
import lotvol_model as lv


def rel_err(X1, X2):
    """
    Relative error between two JAX arrays.

    Adds 0.1 to the denominator to avoid nan's when its equal to zero.
    """
    x1 = X1.ravel() * 1.0
    x2 = X2.ravel() * 1.0
    return jnp.max(jnp.abs((x1 - x2)/(0.1 + x1)))


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
    self.Model = pf.BMModel


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
    self.n_particles = 15
    self.Model = pf.LotVolModel
    self.Model2 = lv.LotVolModel


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
    y_meas1, x_state1 = pf.simulate_for(
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
    pf_out1 = pf.particle_filter_for(model, subkey,
                                     y_meas, theta, n_particles)
    # pf without for-loop
    pf_out2 = pf.particle_filter(
        model, subkey, y_meas, theta, n_particles)
    for k in pf_out1.keys():
        with self.subTest(k=k):
            self.assertAlmostEqual(rel_err(pf_out1[k], pf_out2[k]), 0.0)


def test_for_pf_mvn (self):
    """ particle filter with mvn """
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
    pf_out1 = pf.particle_filter(
        model, subkey, y_meas, theta, n_particles,
        particle_sampler = pf.particle_resample_mvn_for)
    # pf without for-loop
    pf_out2 = pf.particle_filter(
        model, subkey, y_meas, theta, n_particles,
        particle_sampler = pf.particle_resample_mvn)
    for k in pf_out1.keys():
        with self.subTest(k=k):
            self.assertAlmostEqual(rel_err(pf_out1[k], pf_out2[k]), 0.0)


def test_for_mvn_resampler (self):
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
    print(x_particles.shape)
    new_particles_for = pf.particle_resample_mvn_for(
        subkey,
        x_particles,
        logw)
    new_particles = pf.particle_resample_mvn(
        subkey,
        x_particles,
        logw)
    for k in new_particles.keys():
        with self.subTest(k=k):
            self.assertAlmostEqual(rel_err(new_particles[k], new_particles_for[k]), 0.0)
    


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
    x_state1 = pf.particle_smooth_for(
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
    loglik1 = pf.full_loglik_for(model,
                                 y_meas, x_state, theta)
    # joint loglikelihood with vmap
    loglik2 = pf.full_loglik(model,
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
    mwg_out1 = mcmc.param_mwg_update_for(model, prior, subkey, theta,
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
    mwg_out1 = mcmc.param_mwg_update_for(model, prior, subkey, theta,
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


def test_jit_pf_mvn (self):
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
        particle_sampler = pf.particle_resample_mvn)
    # pf with jit
    pf_out2 = jax.jit(pf.particle_filter, static_argnums=(0, 4, 5))(
        model, subkey, y_meas, theta, n_particles,
        particle_sampler = pf.particle_resample_mvn)
    for k in pf_out1.keys():
        with self.subTest(k=k):
            self.assertAlmostEqual(rel_err(pf_out1[k], pf_out2[k]), 0.0)
    # objective function for gradient
    def obj_fun(model, key, y_meas, theta, n_particles):
        return pf.particle_loglik(pf.particle_filter(
            model, key, y_meas, theta, n_particles,
            particle_sampler = pf.particle_resample_mvn)["logw"])

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
    loglik1 = pf.full_loglik(model,
                             y_meas, x_state, theta)
    # joint loglikelihood with jit
    full_loglik_jit = jax.jit(pf.full_loglik, static_argnums=0)
    loglik2 = full_loglik_jit(model,
                              y_meas, x_state, theta)
    self.assertAlmostEqual(rel_err(loglik1, loglik2), 0.0)
    # grad without jit
    grad1 = jax.grad(pf.full_loglik, argnums=(2, 3))(
        model, y_meas, x_state, theta)
    # grad with jit
    grad2 = jax.jit(jax.grad(pf.full_loglik, argnums=(2, 3)),
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
    loglik1 = pf.full_loglik(model1,
                             y_meas, x_state, theta)
    # joint loglikelihood with inherited class
    loglik2 = pf.full_loglik(model2,
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

