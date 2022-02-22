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
import pfjax.mcmc
from utils import *


# # hack to copy-paste in contents without import
# exec(open("bm_model.py").read())
# exec(open("particle_filter.py").read())

# # global variable (can't be defined inside class...)
# dt = .1

def test_setup():
    """
    Creates input arguments to tests.

    Use this instead of TestCase.setUp because I don't want to prefix every variable by `self`.
    """
    key = random.PRNGKey(0)
    # parameter values
    mu = 5
    sigma = 1
    tau = .1
    theta = jnp.array([mu, sigma, tau])
    # data specification
    dt = .1
    n_obs = 5
    x_init = jnp.array([0.])
    return key, theta, dt, n_obs, x_init


def lotvol_test_setup():
    """
    Creates input arguments to tests for lotvol model.
    """
    key = random.PRNGKey(0)
    # parameter values
    alpha = 1.0
    beta = 1.0
    gamma = 4.0
    delta = 1.0
    sigma_h = 0.1
    sigma_l = 0.1
    tau_h = 0.25  # low noise = 0.1
    tau_l = 0.25  # low noise = 0.1
    theta = np.array([alpha, beta, gamma, delta, sigma_h, sigma_l, tau_h, tau_l])
    dt = 0.1
    n_res = 5
    n_obs = 50
    x_init = jnp.block([[jnp.zeros((n_res-1, 2))],
                        [jnp.log(jnp.array([5., 3.]))]])
    return key, theta, dt, n_obs, n_res, x_init

class TestFor(unittest.TestCase):
    """
    Test whether for-loop version of functions is identical to xmap/scan version.
    """

    def test_sim(self):
        key, theta, dt, n_obs, x_init = test_setup()
        model = pf.BMModel(dt=dt)
        # simulate with for-loop
        y_meas1, x_state1 = pf.simulate_for(
            model, key, n_obs, x_init, theta)
        # simulate without for-loop
        y_meas2, x_state2 = pf.simulate(model, key, n_obs, x_init, theta)
        self.assertAlmostEqual(rel_err(y_meas1, y_meas2), 0.0)
        self.assertAlmostEqual(rel_err(x_state1, x_state2), 0.0)

    def test_pf(self):
        key, theta, dt, n_obs, x_init = test_setup()
        model = pf.BMModel(dt=dt)
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
        key, theta, dt, n_obs, x_init = test_setup()
        model = pf.BMModel(dt=dt)
        # simulate without for-loop
        y_meas, x_state = pf.simulate(model, key, n_obs, x_init, theta)
        # joint loglikelihood with for-loop
        loglik1 = pf.mcmc.full_loglik_for(model,
                                          y_meas, x_state, theta)
        # joint loglikelihood with vmap
        loglik2 = pf.mcmc.full_loglik(model,
                                      y_meas, x_state, theta)
        self.assertAlmostEqual(rel_err(loglik1, loglik2), 0.0)


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


class TestMVN(unittest.TestCase):
    """
    Test whether for-loop version of MVN sampling function is identical to xmap/scan version.
    """

    # def BM_test_pf_mvn(self):
    #     key, theta, dt, n_obs, x_init = test_setup()
    #     model = pf.BMModel(dt=dt)
    #     # simulate without for-loop
    #     y_meas, x_state = pf.simulate(model, key, n_obs, x_init, theta)
    #     # particle filter specification
    #     n_particles = 7
    #     key, subkey = random.split(key)
    #     # pf with for-loop MVN sampling
    #     pf_out1 = pf.particle_filter(
    #         model, 
    #         subkey,
    #         y_meas, 
    #         theta, 
    #         n_particles,
    #         particle_sampler=pf.particle_resample_mvn_for)
    #     # pf without for-loop for MVN sampling
    #     pf_out2 = pf.particle_filter(
    #         model,
    #         subkey, 
    #         y_meas, 
    #         theta, 
    #         n_particles,
    #         particle_sampler = pf.particle_resample_mvn)
    #     for k in pf_out1.keys():
    #         with self.subTest(k=k):
    #             self.assertAlmostEqual(rel_err(pf_out1[k], pf_out2[k]), 0.0)

    def lotvol_test_pf_mvn(self):
        key, theta, dt, n_obs, n_res, x_init = lotvol_test_setup()
        model = pf.LotVolModel(dt, n_res)
        # simulate without for-loop
        y_meas, x_state = pf.simulate(model, key, n_obs, x_init, theta)
        # particle filter specification
        n_particles = 10
        key, subkey = random.split(key)
        # pf with for-loop MVN sampling
        pf_out1 = pf.particle_filter(
            model, 
            subkey,
            y_meas, 
            theta, 
            n_particles,
            particle_sampler=pf.particle_resample_mvn_for)
        # pf without for-loop for MVN sampling
        pf_out2 = pf.particle_filter(
            model,
            subkey, 
            y_meas, 
            theta, 
            n_particles,
            particle_sampler = pf.particle_resample_mvn)
        print(pf_out2.keys())
        print(pf_out1.keys())
        for k in pf_out1.keys():
            with self.subTest(k=k):
                self.assertAlmostEqual(rel_err(pf_out1[k], pf_out2[k]), 0.0)




class TestJit(unittest.TestCase):
    """
    Check whether jit with and without grad gives the same result.
    """

    def test_sim(self):
        key, theta, dt, n_obs, x_init = test_setup()
        model = pf.BMModel(dt=dt)
        # simulate without jit
        y_meas1, x_state1 = pf.simulate(model, key, n_obs, x_init, theta)
        # simulate with jit
        simulate_jit = jax.jit(pf.simulate, static_argnums=(0, 2))
        y_meas2, x_state2 = simulate_jit(model, key, n_obs, x_init, theta)
        # # use wrong dt
        # model2 = pf.BMModel(dt=2.0 * dt)
        # y_meas2, x_state2 = simulate_jit(model2, n_obs, x_init, theta, key)
        # breakpoint()
        # # use correct dt
        # model2.dt = dt
        # y_meas2, x_state2 = simulate_jit(model2, n_obs, x_init, theta, key)
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
        key, theta, dt, n_obs, x_init = test_setup()
        # simulate data
        model = pf.BMModel(dt=dt)
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
        key, theta, dt, n_obs, x_init = test_setup()
        # simulate data
        model = pf.BMModel(dt=dt)
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
