"""
Unit tests for the particle filter.

Things to test:

- [ ] `jit` + `grad` return without errors.
- [ ] `vmap`, `xmap`, `scan`, etc. give the same result as with for-loops.
- [ ] Global and OOP APIs give the same results.
- [ ] OOP API treats class members as expected, i.e., not like using globals in jitted functions.
"""

import unittest
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import particle_filter as pf
import bm_model as bm


def rel_err(X1, X2):
    """
    Relative error between two JAX arrays.

    Adds a bit to the denominator when it's equal to zero.
    """
    return jnp.max(jnp.abs((X1.ravel() - X2.ravel())/(0.1 + X1.ravel())))


# hack to copy-paste in contents without import
exec(open("tests/bm_model.py").read())
exec(open("tests/particle_filter.py").read())

# global variable (can't be defined inside class...)
dt = .1


class TestFor(unittest.TestCase):
    """
    Test whether for-loop version of functions is identical to xmap/scan version.
    """

    def test_sim(self):
        key = random.PRNGKey(0)
        # parameter values
        mu = 5
        sigma = 1
        tau = .1
        theta = jnp.array([mu, sigma, tau])
        # data specification
        n_obs = 5
        x_init = jnp.array([0.])
        bm_model = bm.BMFilter(dt=dt)
        # simulate with for-loop
        y_meas1, x_state1 = bm_model.meas_sim_for(n_obs, x_init, theta, key)
        # simulate without for-loop
        y_meas2, x_state2 = bm_model.meas_sim(n_obs, x_init, theta, key)
        self.assertAlmostEqual(rel_err(y_meas1, y_meas2), 0.0)
        self.assertAlmostEqual(rel_err(x_state1, x_state2), 0.0)

    def test_pf(self):
        key = random.PRNGKey(0)
        # parameter values
        mu = 5
        sigma = 1
        tau = .1
        theta = jnp.array([mu, sigma, tau])
        # data specification
        n_obs = 5
        x_init = jnp.array([0.])
        bm_model = bm.BMFilter(dt=dt)
        # simulate without for-loop
        y_meas, x_state = bm_model.meas_sim(n_obs, x_init, theta, key)
        # run particle filter
        n_particles = 7
        key, subkey = random.split(key)
        pf_out1 = bm_model.particle_filter_for(
            y_meas, theta, n_particles, subkey)
        pf_out2 = bm_model.particle_filter(y_meas, theta, n_particles, subkey)
        for k in pf_out1.keys():
            with self.subTest(k=k):
                self.assertAlmostEqual(rel_err(pf_out1[k], pf_out2[k]), 0.0)


class TestOOP(unittest.TestCase):
    """
    Test whether purely functional API and OOP API give the same resuls.
    """

    def test_sim(self):
        key = random.PRNGKey(0)
        # parameter values
        mu = 5
        sigma = 1
        tau = .1
        theta = jnp.array([mu, sigma, tau])
        # data specification
        n_obs = 5
        x_init = jnp.array([0.])
        # simulate with globals
        y_meas1, x_state1 = meas_sim(n_obs, x_init, theta, key)
        # simulate with oop
        bm_model = bm.BMFilter(dt=dt)
        y_meas2, x_state2 = bm_model.meas_sim(n_obs, x_init, theta, key)
        self.assertAlmostEqual(rel_err(y_meas1, y_meas2), 0.0)
        self.assertAlmostEqual(rel_err(x_state1, x_state2), 0.0)

    def test_pf(self):
        key = random.PRNGKey(0)
        # parameter values
        mu = 5
        sigma = 1
        tau = .1
        theta = jnp.array([mu, sigma, tau])
        # data specification
        n_obs = 5
        x_init = jnp.array([0.])
        # simulate with oop
        bm_model = bm.BMFilter(dt=dt)
        y_meas, x_state = bm_model.meas_sim(n_obs, x_init, theta, key)
        # run particle filter
        n_particles = 7
        key, subkey = random.split(key)
        pf_out1 = particle_filter(y_meas, theta, n_particles, subkey)
        pf_out2 = bm_model.particle_filter(y_meas, theta, n_particles, subkey)
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
        mu = 5
        sigma = 1
        tau = .1
        theta = jnp.array([mu, sigma, tau])
        # data specification
        n_obs = 5
        x_init = jnp.array([0.])
        # simulate without jit
        bm_model = bm.BMFilter(dt=dt)
        y_meas1, x_state1 = bm_model.meas_sim(n_obs, x_init, theta, key)
        # simulate with jit
        y_meas2, x_state2 = jax.jit(bm_model.meas_sim, static_argnums=0)(
            n_obs, x_init, theta, key)
        self.assertAlmostEqual(rel_err(y_meas1, y_meas2), 0.0)
        self.assertAlmostEqual(rel_err(x_state1, x_state2), 0.0)

    def test_pf(self):
        key = random.PRNGKey(0)
        # parameter values
        mu = 5
        sigma = 1
        tau = .1
        theta = jnp.array([mu, sigma, tau])
        # data specification
        n_obs = 5
        x_init = jnp.array([0.])
        # simulate data
        bm_model = bm.BMFilter(dt=dt)
        y_meas, x_state = bm_model.meas_sim(n_obs, x_init, theta, key)
        # run particle filter
        n_particles = 7
        key, subkey = random.split(key)
        pf_out1 = bm_model.particle_filter(y_meas, theta, n_particles, subkey)
        pf_out2 = jax.jit(bm_model.particle_filter, static_argnums=2)(
            y_meas, theta, n_particles, subkey)
        for k in pf_out1.keys():
            with self.subTest(k=k):
                self.assertAlmostEqual(rel_err(pf_out1[k], pf_out2[k]), 0.0)


if __name__ == '__main__':
    unittest.main()

# --- scratch ------------------------------------------------------------------


# key = random.PRNGKey(0)


# # parameter values
# mu = 5
# sigma = 1
# tau = .1
# theta = jnp.array([mu, sigma, tau])

# print(theta)

# # data specification
# dt = .1
# n_obs = 5
# x_init = jnp.array([0.])

# # simulate regular data
# y_meas_for, x_state_for = meas_sim_for(n_obs, x_init, theta, key)

# # simulate lax data
# y_meas, x_state = meas_sim(n_obs, x_init, theta, key)

# print("max_diff between sim_for and sim_opt:\n")
# print("y_meas = \n",
#       jnp.max(jnp.abs(y_meas_for - y_meas)))
# print("x_state = \n",
#       jnp.max(jnp.abs(x_state_for - x_state)))

# # run particle filter
# n_particles = 7
# key, subkey = random.split(key)
# pf_for = particle_filter_for(y_meas, theta, n_particles, subkey)
# pf_out = particle_filter(y_meas, theta, n_particles, subkey)

# print("max_diff between pf_for and pf_opt:\n")
# for k in pf_for.keys():
#     print(k, " = \n",
#           jnp.max(jnp.abs(pf_for[k] - pf_out[k])))
