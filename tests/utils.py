import unittest
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import pfjax as pf
import pfjax.mcmc


def rel_err(X1, X2):
    """
    Relative error between two JAX arrays.

    Adds 0.1 to the denominator to avoid nan's when its equal to zero.
    """
    return jnp.max(jnp.abs((X1.ravel() - X2.ravel())/(0.1 + X1.ravel())))


# --- now some generic external methods for constructing the tests... ----------


class TestForBase(unittest.TestCase):
    """
    Test whether for-loop version of functions is identical to xmap/scan version.
    """

    def test_sim(self):
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

    def test_pf(self):
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
        # particle filter specification
        n_particles = 7
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

    def test_loglik(self):
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
        loglik1 = pf.mcmc.full_loglik_for(model,
                                          y_meas, x_state, theta)
        # joint loglikelihood with vmap
        loglik2 = pf.mcmc.full_loglik(model,
                                      y_meas, x_state, theta)
        self.assertAlmostEqual(rel_err(loglik1, loglik2), 0.0)


class TestJitBase(unittest.TestCase):
    """
    Check whether jit with and without grad gives the same result.
    """

    def test_sim(self):
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

    def test_pf(self):
        # un-self setUp members
        key = self.key
        theta = self.theta
        x_init = self.x_init
        model_args = self.model_args
        n_obs = self.n_obs
        n_particles = self.n_particles
        model = self.Model(**model_args)
        # simulate data
        y_meas, x_state = pf.simulate(model, key, n_obs, x_init, theta)
        # particle filter specification
        n_particles = 7
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

    def test_loglik(self):
        # un-self setUp members
        key = self.key
        theta = self.theta
        x_init = self.x_init
        model_args = self.model_args
        n_obs = self.n_obs
        n_particles = self.n_particles
        model = self.Model(**model_args)
        # simulate data
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


class TestModelsBase(unittest.TestCase):
    """
    Test that two model definitions are equivalent.
    """

    def test_sim(self):
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

    def test_loglik(self):
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
        loglik1 = pf.mcmc.full_loglik(model1,
                                      y_meas, x_state, theta)
        # joint loglikelihood with inherited class
        loglik2 = pf.mcmc.full_loglik(model2,
                                      y_meas, x_state, theta)
        self.assertAlmostEqual(rel_err(loglik1, loglik2), 0.0)

    def test_pf(self):
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


def test_for_sim(self, Model, test_setup):
    key, theta, x_init, model_args, n_obs, n_particles = test_setup()
    model = Model(**model_args)
    # simulate with for-loop
    y_meas1, x_state1 = pf.simulate_for(
        model, key, n_obs, x_init, theta)
    # simulate without for-loop
    y_meas2, x_state2 = pf.simulate(model, key, n_obs, x_init, theta)
    self.assertAlmostEqual(rel_err(y_meas1, y_meas2), 0.0)
    self.assertAlmostEqual(rel_err(x_state1, x_state2), 0.0)
