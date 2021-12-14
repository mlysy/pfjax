"""
Unit tests for the particle filter.

Things to test:

- [ ] `jit` + `grad` return without errors.
- [ ] `vmap`, `xmap`, `scan`, etc. give the same result as with for-loops.
- [ ] Global and OOP APIs give the same results.
- [ ] OOP API treats class members as expected, i.e., not like using globals in jitted functions.

Test code: from `pfjax` top directory:

```
python -m unittest tests/test_*
```

**FIXME:** This doesn't work well because the library hasn't been set up as a proper Python package.  Hopefully can fix imports etc. to run this properly once the package is installed.
"""

import unittest
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
from .utils import *  # for running tests from top-level directory
import particle_filter as pf
import bm_model as bm


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
        bm_model = bm.BMModel(dt=dt)
        # simulate with for-loop
        y_meas1, x_state1 = pf.meas_sim_for(
            bm_model, n_obs, x_init, theta, key)
        # simulate without for-loop
        y_meas2, x_state2 = pf.meas_sim(bm_model, n_obs, x_init, theta, key)
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
        bm_model = bm.BMModel(dt=dt)
        # simulate without for-loop
        y_meas, x_state = pf.meas_sim(bm_model, n_obs, x_init, theta, key)
        # particle filter specification
        n_particles = 7
        key, subkey = random.split(key)
        # pf with for-loop
        pf_out1 = pf.particle_filter_for(bm_model,
                                         y_meas, theta, n_particles, subkey)
        # pf without for-loop
        pf_out2 = pf.particle_filter(
            bm_model, y_meas, theta, n_particles, subkey)
        for k in pf_out1.keys():
            with self.subTest(k=k):
                self.assertAlmostEqual(rel_err(pf_out1[k], pf_out2[k]), 0.0)

    def test_loglik(self):
        key = random.PRNGKey(0)
        # parameter values
        mu = 5
        sigma = 1
        tau = .1
        theta = jnp.array([mu, sigma, tau])
        # data specification
        n_obs = 10
        x_init = jnp.array([0.])
        bm_model = bm.BMModel(dt=dt)
        # simulate without for-loop
        y_meas, x_state = pf.meas_sim(bm_model, n_obs, x_init, theta, key)
        # joint loglikelihood with for-loop
        loglik1 = pf.joint_loglik_for(bm_model,
                                      y_meas, x_state, theta)
        # joint loglikelihood with vmap
        loglik2 = pf.joint_loglik(bm_model,
                                  y_meas, x_state, theta)
        self.assertAlmostEqual(rel_err(loglik1, loglik2), 0.0)


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
        bm_model = bm.BMModel(dt=dt)
        y_meas2, x_state2 = pf.meas_sim(bm_model, n_obs, x_init, theta, key)
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
        bm_model = bm.BMModel(dt=dt)
        y_meas, x_state = pf.meas_sim(bm_model, n_obs, x_init, theta, key)
        # particle filter specification
        n_particles = 7
        key, subkey = random.split(key)
        # pf with globals
        pf_out1 = particle_filter(y_meas, theta, n_particles, subkey)
        # pf with oop
        pf_out2 = pf.particle_filter(
            bm_model, y_meas, theta, n_particles, subkey)
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
        bm_model = bm.BMModel(dt=dt)
        y_meas1, x_state1 = pf.meas_sim(bm_model, n_obs, x_init, theta, key)
        # simulate with jit
        meas_sim_jit = jax.jit(pf.meas_sim, static_argnums=(0, 1))
        bm_model2 = bm.BMModel(dt=dt)
        y_meas2, x_state2 = meas_sim_jit(bm_model2, n_obs, x_init, theta, key)
        # # use wrong dt
        # bm_model2 = bm.BMModel(dt=2.0 * dt)
        # y_meas2, x_state2 = meas_sim_jit(bm_model2, n_obs, x_init, theta, key)
        # breakpoint()
        # # use correct dt
        # bm_model2.dt = dt
        # y_meas2, x_state2 = meas_sim_jit(bm_model2, n_obs, x_init, theta, key)
        self.assertAlmostEqual(rel_err(y_meas1, y_meas2), 0.0)
        self.assertAlmostEqual(rel_err(x_state1, x_state2), 0.0)
        # objective function for gradient
        def obj_fun(model, n_obs, x_init, theta, key): return jnp.mean(
            pf.meas_sim(model, n_obs, x_init, theta, key)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun, argnums=3)(
            bm_model, n_obs, x_init, theta, key)
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun, argnums=3), static_argnums=(0, 1))(
            bm_model, n_obs, x_init, theta, key)
        self.assertAlmostEqual(rel_err(grad1, grad2), 0.0)

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
        bm_model = bm.BMModel(dt=dt)
        y_meas, x_state = pf.meas_sim(bm_model, n_obs, x_init, theta, key)
        # particle filter specification
        n_particles = 7
        key, subkey = random.split(key)
        # pf without jit
        pf_out1 = pf.particle_filter(
            bm_model, y_meas, theta, n_particles, subkey)
        # pf with jit
        pf_out2 = jax.jit(pf.particle_filter, static_argnums=(0, 3))(
            bm_model, y_meas, theta, n_particles, subkey)
        for k in pf_out1.keys():
            with self.subTest(k=k):
                self.assertAlmostEqual(rel_err(pf_out1[k], pf_out2[k]), 0.0)

        # objective function for gradient
        def obj_fun(model, y_meas, theta, n_particles, key):
            return pf.particle_loglik(pf.particle_filter(
                model, y_meas, theta, n_particles, key)["logw_particles"])
        # grad without jit
        grad1 = jax.grad(obj_fun, argnums=2)(
            bm_model, y_meas, theta, n_particles, key)
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun, argnums=2), static_argnums=(0, 3))(
            bm_model, y_meas, theta, n_particles, key)
        self.assertAlmostEqual(rel_err(grad1, grad2), 0.0)


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

# # particle filter specification
# n_particles = 7
# key, subkey = random.split(key)
# pf_for = particle_filter_for(y_meas, theta, n_particles, subkey)
# pf_out = particle_filter(y_meas, theta, n_particles, subkey)

# print("max_diff between pf_for and pf_opt:\n")
# for k in pf_for.keys():
#     print(k, " = \n",
#           jnp.max(jnp.abs(pf_for[k] - pf_out[k])))
