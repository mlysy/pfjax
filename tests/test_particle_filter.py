"""
Unit tests for the particle filter.

Things to test:

- [x] `jit` + `grad` return without errors.

- [x] `vmap`, `xmap`, `scan`, etc. give the same result as with for-loops.

- [x] Global and OOP APIs give the same results.

    **Deprecated:** Too cumbersome to maintain essentially duplicate APIs. 

- [x] OOP API treats class members as expected, i.e., not like using globals in jitted functions.

    **Update:** Does in fact treat data members as globals, unless the class is made hashable.


Test code: from `pfjax/tests`:

```
python -m unittest -v
```
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import pfjax as pf
import pfjax.mcmc as mcmc
import utils


class TestFor(unittest.TestCase):
    """
    Test whether for-loop version of functions is identical to xmap/scan version.
    """

    setUp = utils.bm_setup

    test_sim = utils.test_for_simulate
    test_loglik = utils.test_for_loglik_full
    test_pf_multinomial = utils.test_for_particle_filter
    test_pf_mvn = utils.test_for_resample_mvn
    test_pf_mvn_shape = utils.test_shape_resample_mvn
    test_smooth = utils.test_for_particle_smooth


class TestJit(unittest.TestCase):
    """
    Check whether jit with and without grad gives the same result.
    """

    setUp = utils.bm_setup

    test_sim = utils.test_jit_simulate
    test_loglik = utils.test_jit_loglik_full
    test_pf = utils.test_jit_particle_filter
    test_pf_mvn = utils.test_jit_resample_mvn
    test_smooth = utils.test_jit_particle_smooth


if __name__ == '__main__':
    unittest.main()

# --- scratch ------------------------------------------------------------------

# def test_setup(self):
#     """
#     Creates input arguments to tests.

#     Use this instead of TestCase.setUp because I don't want to prefix every variable by `self`.
#     """
#     key = random.PRNGKey(0)
#     # parameter values
#     mu = 5
#     sigma = 1
#     tau = .1
#     theta = jnp.array([mu, sigma, tau])
#     # data specification
#     dt = .1
#     n_obs = 5
#     x_init = jnp.array([0.])
#     # particle filter specification
#     n_particles = 7
#     return key, theta, x_init, {"dt": dt}, n_obs, n_particles


# class TestOOP(unittest.TestCase):
#     """
#     Test whether purely functional API and OOP API give the same resuls.
#     """

#     def test_sim(self):
#         key = random.PRNGKey(0)
#         # parameter values
#         mu = 5
#         sigma = 1
#         tau = .1
#         theta = jnp.array([mu, sigma, tau])
#         # data specification
#         n_obs = 5
#         x_init = jnp.array([0.])
#         # simulate with globals
#         y_meas1, x_state1 = simulate(n_obs, x_init, theta, key)
#         # simulate with oop
#         model = pf.BMModel(dt=dt)
#         y_meas2, x_state2 = pf.simulate(model, n_obs, x_init, theta, key)
#         self.assertAlmostEqual(rel_err(y_meas1, y_meas2), 0.0)
#         self.assertAlmostEqual(rel_err(x_state1, x_state2), 0.0)

#     def test_pf(self):
#         key = random.PRNGKey(0)
#         # parameter values
#         mu = 5
#         sigma = 1
#         tau = .1
#         theta = jnp.array([mu, sigma, tau])
#         # data specification
#         n_obs = 5
#         x_init = jnp.array([0.])
#         # simulate with oop
#         model = pf.BMModel(dt=dt)
#         y_meas, x_state = pf.simulate(model, n_obs, x_init, theta, key)
#         # particle filter specification
#         n_particles = 7
#         key, subkey = random.split(key)
#         # pf with globals
#         pf_out1 = particle_filter(y_meas, theta, n_particles, subkey)
#         # pf with oop
#         pf_out2 = pf.particle_filter(
#             model, y_meas, theta, n_particles,
#             pf.particle_resample, subkey)
#         for k in pf_out1.keys():
#             with self.subTest(k=k):
#                 self.assertAlmostEqual(rel_err(pf_out1[k], pf_out2[k]), 0.0)


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
# y_meas_for, x_state_for = simulate_for(n_obs, x_init, theta, key)

# # simulate lax data
# y_meas, x_state = simulate(n_obs, x_init, theta, key)

# print("max_diff between sim_for and sim_opt:\n")
# print("y_meas = \n",
#       jnp.max(jnp.abs(y_meas_for - y_meas)))
# print("x_state = \n",
#       jnp.max(jnp.abs(x_state_for - x_state)))

# # particle filter specification
# n_particles = 7
# key, subkey = random.split(key)
# pf_for = particle_filter_for(y_meas, theta, n_particles, subkey)
# pf_out = particle_filter(y_meas, theta, n_particles, subkey)

# print("max_diff between pf_for and pf_opt:\n")
# for k in pf_for.keys():
#     print(k, " = \n",
#           jnp.max(jnp.abs(pf_for[k] - pf_out[k])))
