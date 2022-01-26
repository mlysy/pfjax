"""
Unit tests for SDE methods.

Things to test:

- [ ] SDE base class works as expected, i.e., switches between `euler_{sim/lpdf}_diag()` and `euler_{sim/lpdf}_var()` at instantiation.

- [x] `jit` + `grad` return without errors.

- [x] JAX constructs (e.g., `vmap`, `xmap`, `lax.scan`, etc.) give the same result as for-loops, etc.

"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import pfjax as pf
import pfjax.mcmc as mcmc
import lotvol_model as lv
from utils import *


def test_setup():
    """
    Creates input arguments to tests.

    Use this instead of TestCase.setUp because I don't want to prefix every variable by `self`.
    """
    key = random.PRNGKey(0)
    # parameter values
    alpha = 1.02
    beta = 1.02
    gamma = 4.
    delta = 1.04
    sigma_H = .1
    sigma_L = .2
    tau_H = .25
    tau_L = .35
    theta = jnp.array([alpha, beta, gamma, delta,
                       sigma_H, sigma_L, tau_H, tau_L])
    # data specification
    dt = .09
    n_res = 3
    n_obs = 7
    x_init = jnp.block([[jnp.zeros((n_res-1, 2))],
                        [jnp.log(jnp.array([5., 3.]))]])
    return key, theta, dt, n_res, n_obs, x_init


class TestInherit(unittest.TestCase):
    """
    Test that we can inherit from SDE base class as expected.
    """

    def test_sim(self):
        key, theta, dt, n_res, n_obs, x_init = test_setup()
        # simulate with non-inherited class
        model1 = lv.LotVolModel(dt=dt, n_res=n_res)
        y_meas1, x_state1 = pf.simulate(model1, key, n_obs, x_init, theta)
        # simulate with inherited class
        model2 = pf.LotVolModel(dt=dt, n_res=n_res)
        y_meas2, x_state2 = pf.simulate(model2, key, n_obs, x_init, theta)
        self.assertAlmostEqual(rel_err(y_meas1, y_meas2), 0.0)
        self.assertAlmostEqual(rel_err(x_state1, x_state2), 0.0)

    def test_loglik(self):
        key, theta, dt, n_res, n_obs, x_init = test_setup()
        # model with non-inherited class
        model1 = lv.LotVolModel(dt=dt, n_res=n_res)
        # model with inherited class
        model2 = pf.LotVolModel(dt=dt, n_res=n_res)
        # simulate with inherited class
        y_meas, x_state = pf.simulate(model2, key, n_obs, x_init, theta)
        # joint loglikelihood with non-inherited class
        loglik1 = pf.mcmc.full_loglik(model1,
                                      y_meas, x_state, theta)
        # joint loglikelihood with inherited class
        loglik2 = pf.mcmc.full_loglik(model2,
                                      y_meas, x_state, theta)
        self.assertAlmostEqual(rel_err(loglik1, loglik2), 0.0)

    def test_pf(self):
        key, theta, dt, n_res, n_obs, x_init = test_setup()
        # model with non-inherited class
        model1 = lv.LotVolModel(dt=dt, n_res=n_res)
        # model with inherited class
        model2 = pf.LotVolModel(dt=dt, n_res=n_res)
        # simulate with inherited class
        y_meas, x_state = pf.simulate(model2, key, n_obs, x_init, theta)
        # particle filter specification
        n_particles = 2
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


class TestJit(unittest.TestCase):
    """
    Check whether jit with and without grad gives the same result.
    """

    def test_sim(self):
        key, theta, dt, n_res, n_obs, x_init = test_setup()
        model = pf.LotVolModel(dt=dt, n_res=n_res)
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

    def test_loglik(self):
        key, theta, dt, n_res, n_obs, x_init = test_setup()
        # simulate data
        model = pf.LotVolModel(dt=dt, n_res=n_res)
        y_meas, x_state = pf.simulate(model, key, n_obs, x_init, theta)
        # joint loglikelihood without jit
        loglik1 = pf.mcmc.full_loglik(model,
                                      y_meas, x_state, theta)
        # joint loglikelihood with jit
        full_loglik_jit = jax.jit(pf.mcmc.full_loglik, static_argnums=0)
        loglik2 = full_loglik_jit(model,
                                  y_meas, x_state, theta)
        self.assertAlmostEqual(rel_err(loglik1, loglik2), 0.0)
        # grad without jit
        grad1 = jax.grad(pf.mcmc.full_loglik, argnums=(2, 3))(
            model, y_meas, x_state, theta)
        # grad with jit
        grad2 = jax.jit(jax.grad(pf.mcmc.full_loglik, argnums=(2, 3)),
                        static_argnums=0)(model, y_meas, x_state, theta)
        for i in range(2):
            with self.subTest(i=i):
                self.assertAlmostEqual(rel_err(grad1[i], grad2[i]), 0.0)

    def test_pf(self):
        key, theta, dt, n_res, n_obs, x_init = test_setup()
        # simulate data
        model = pf.LotVolModel(dt=dt, n_res=n_res)
        y_meas, x_state = pf.simulate(model, key, n_obs, x_init, theta)
        # particle filter specification
        n_particles = 2
        key, subkey = random.split(key)
        # pf without jit
        pf_out1 = pf.particle_filter(
            model, subkey, y_meas, theta, n_particles)
        # pf with jit
        particle_filter_jit = jax.jit(
            pf.particle_filter, static_argnums=(0, 4))
        pf_out2 = particle_filter_jit(
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


class TestFor(unittest.TestCase):
    """
    Test whether for-loop version of functions is identical to xmap/scan version.
    """

    def test_state_sample(self):
        key, theta, dt, n_res, n_obs, x_init = test_setup()
        model = pf.LotVolModel(dt=dt, n_res=n_res)
        # generate previous timepoint
        key, subkey = random.split(key)
        x_prev = jnp.block([[jnp.zeros((n_res-1, 2))],
                            [jnp.log(jnp.array([5., 3.]))]])
        x_prev = x_prev + random.normal(subkey, x_prev.shape)
        # simulate state using for-loop
        x_state1 = model.state_sample_for(key, x_prev, theta)
        # simulate state using lax.scan
        x_state2 = model.state_sample(key, x_prev, theta)
        self.assertAlmostEqual(rel_err(x_state1, x_state2), 0.0)

    def test_state_lpdf(self):
        key, theta, dt, n_res, n_obs, x_init = test_setup()
        model = pf.LotVolModel(dt=dt, n_res=n_res)
        # generate previous timepoint
        key, subkey = random.split(key)
        x_prev = jnp.block([[jnp.zeros((n_res-1, 2))],
                            [jnp.log(jnp.array([5., 3.]))]])
        x_prev = x_prev + random.normal(subkey, x_prev.shape)
        # simulate state using lax.scan
        x_curr = model.state_sample(key, x_prev, theta)
        # lpdf using for
        lp1 = model.state_lpdf_for(x_curr, x_prev, theta)
        lp2 = model.state_lpdf(x_curr, x_prev, theta)
        self.assertAlmostEqual(rel_err(lp1, lp2), 0.0)
