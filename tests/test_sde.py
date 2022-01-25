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


class TestInherit(unittest.TestCase):
    """
    Test that we can inherit from SDE base class as expected.
    """

    def test_sim(self):
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
        # simulate with non-inherited class
        lv_model1 = lv.LotVolModel(dt=dt, n_res=n_res)
        y_meas1, x_state1 = pf.simulate(lv_model1, n_obs, x_init, theta, key)
        # simulate with inherited class
        lv_model2 = pf.LotVolModel(dt=dt, n_res=n_res)
        y_meas2, x_state2 = pf.simulate(lv_model2, n_obs, x_init, theta, key)
        self.assertAlmostEqual(rel_err(y_meas1, y_meas2), 0.0)
        self.assertAlmostEqual(rel_err(x_state1, x_state2), 0.0)

    def test_loglik(self):
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
        # model with non-inherited class
        lv_model1 = lv.LotVolModel(dt=dt, n_res=n_res)
        # model with inherited class
        lv_model2 = pf.LotVolModel(dt=dt, n_res=n_res)
        # simulate with inherited class
        y_meas, x_state = pf.simulate(lv_model2, n_obs, x_init, theta, key)
        # joint loglikelihood with non-inherited class
        loglik1 = pf.mcmc.full_loglik(lv_model1,
                                      y_meas, x_state, theta)
        # joint loglikelihood with inherited class
        loglik2 = pf.mcmc.full_loglik(lv_model2,
                                      y_meas, x_state, theta)
        self.assertAlmostEqual(rel_err(loglik1, loglik2), 0.0)

    def test_pf(self):
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
        # model with non-inherited class
        lv_model1 = lv.LotVolModel(dt=dt, n_res=n_res)
        # model with inherited class
        lv_model2 = pf.LotVolModel(dt=dt, n_res=n_res)
        # simulate with inherited class
        y_meas, x_state = pf.simulate(lv_model2, n_obs, x_init, theta, key)
        # particle filter specification
        n_particles = 2
        key, subkey = random.split(key)
        # pf with non-inherited class
        pf_out1 = pf.particle_filter(
            lv_model1, y_meas, theta, n_particles,
            pf.particle_resample, subkey
        )
        # pf with inherited class
        pf_out2 = pf.particle_filter(
            lv_model2, y_meas, theta, n_particles,
            pf.particle_resample, subkey
        )
        for k in pf_out1.keys():
            with self.subTest(k=k):
                self.assertAlmostEqual(rel_err(pf_out1[k], pf_out2[k]), 0.0)


class TestJit(unittest.TestCase):
    """
    Check whether jit with and without grad gives the same result.
    """

    def test_sim(self):
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
        lv_model = pf.LotVolModel(dt=dt, n_res=n_res)
        # simulate without jit
        y_meas1, x_state1 = pf.simulate(lv_model, n_obs, x_init, theta, key)
        # simulate with jit
        simulate_jit = jax.jit(pf.simulate, static_argnums=(0, 1))
        y_meas2, x_state2 = simulate_jit(lv_model, n_obs, x_init, theta, key)
        self.assertAlmostEqual(rel_err(y_meas1, y_meas2), 0.0)
        self.assertAlmostEqual(rel_err(x_state1, x_state2), 0.0)

        # objective function for gradient
        def obj_fun(model, n_obs, x_init, theta, key): return jnp.mean(
            pf.simulate(model, n_obs, x_init, theta, key)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun, argnums=3)(
            lv_model, n_obs, x_init, theta, key)
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun, argnums=3), static_argnums=(0, 1))(
            lv_model, n_obs, x_init, theta, key)
        self.assertAlmostEqual(rel_err(grad1, grad2), 0.0)

    def test_loglik(self):
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
        # simulate data
        lv_model = pf.LotVolModel(dt=dt, n_res=n_res)
        y_meas, x_state = pf.simulate(lv_model, n_obs, x_init, theta, key)
        # joint loglikelihood without jit
        loglik1 = pf.mcmc.full_loglik(lv_model,
                                      y_meas, x_state, theta)
        # joint loglikelihood with jit
        full_loglik_jit = jax.jit(pf.mcmc.full_loglik, static_argnums=0)
        loglik2 = full_loglik_jit(lv_model,
                                  y_meas, x_state, theta)
        self.assertAlmostEqual(rel_err(loglik1, loglik2), 0.0)
        # grad without jit
        grad1 = jax.grad(pf.mcmc.full_loglik, argnums=(2, 3))(
            lv_model, y_meas, x_state, theta)
        # grad with jit
        grad2 = jax.jit(jax.grad(pf.mcmc.full_loglik, argnums=(2, 3)),
                        static_argnums=0)(lv_model, y_meas, x_state, theta)
        for i in range(2):
            with self.subTest(i=i):
                self.assertAlmostEqual(rel_err(grad1[i], grad2[i]), 0.0)

    def test_pf(self):
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
        # simulate data
        lv_model = pf.LotVolModel(dt=dt, n_res=n_res)
        y_meas, x_state = pf.simulate(lv_model, n_obs, x_init, theta, key)
        # particle filter specification
        n_particles = 2
        key, subkey = random.split(key)
        # pf without jit
        pf_out1 = pf.particle_filter(
            lv_model, y_meas, theta, n_particles,
            pf.particle_resample, subkey
        )
        # pf with jit
        particle_filter_jit = jax.jit(
            pf.particle_filter, static_argnums=(0, 3, 4))
        pf_out2 = particle_filter_jit(
            lv_model, y_meas, theta, n_particles,
            pf.particle_resample, subkey
        )
        for k in pf_out1.keys():
            with self.subTest(k=k):
                self.assertAlmostEqual(rel_err(pf_out1[k], pf_out2[k]), 0.0)

        # objective function for gradient
        def obj_fun(model, y_meas, theta, n_particles, key):
            return pf.particle_loglik(pf.particle_filter(
                model, y_meas, theta, n_particles, pf.particle_resample,
                key)["logw"])
        # grad without jit
        grad1 = jax.grad(obj_fun, argnums=2)(
            lv_model, y_meas, theta, n_particles, key)
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun, argnums=2), static_argnums=(0, 3))(
            lv_model, y_meas, theta, n_particles, key)
        self.assertAlmostEqual(rel_err(grad1, grad2), 0.0)


class TestFor(unittest.TestCase):
    """
    Test whether for-loop version of functions is identical to xmap/scan version.
    """

    def test_state_sample(self):
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
        lv_model = pf.LotVolModel(dt=dt, n_res=n_res)
        # generate previous timepoint
        key, subkey = random.split(key)
        x_prev = jnp.block([[jnp.zeros((n_res-1, 2))],
                            [jnp.log(jnp.array([5., 3.]))]])
        x_prev = x_prev + random.normal(subkey, x_prev.shape)
        # simulate state using for-loop
        x_state1 = lv_model.state_sample_for(x_prev, theta, key)
        # simulate state using lax.scan
        x_state2 = lv_model.state_sample(x_prev, theta, key)
        self.assertAlmostEqual(rel_err(x_state1, x_state2), 0.0)

    def test_state_lpdf(self):
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
        lv_model = pf.LotVolModel(dt=dt, n_res=n_res)
        # generate previous timepoint
        key, subkey = random.split(key)
        x_prev = jnp.block([[jnp.zeros((n_res-1, 2))],
                            [jnp.log(jnp.array([5., 3.]))]])
        x_prev = x_prev + random.normal(subkey, x_prev.shape)
        # simulate state using lax.scan
        x_curr = lv_model.state_sample(x_prev, theta, key)
        # lpdf using for
        lp1 = lv_model.state_lpdf_for(x_curr, x_prev, theta)
        lp2 = lv_model.state_lpdf(x_curr, x_prev, theta)
        self.assertAlmostEqual(rel_err(lp1, lp2), 0.0)
