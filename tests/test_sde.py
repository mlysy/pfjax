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
import utils


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


def lv_setup(self):
    """
    Creates input arguments to tests.
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
    self.n_particles = 2
    self.Model = pf.LotVolModel
    self.Model2 = lv.LotVolModel


class TestInherit(utils.TestModelsBase):
    """
    Check that inheritance from SDEModel works as expected.
    """

    setUp = lv_setup


class TestJit(utils.TestJitBase):
    """
    Check whether jit with and without grad gives the same result.
    """

    setUp = lv_setup


class TestFor(utils.TestForBase):
    """
    Test whether for-loop version of functions is identical to xmap/scan version.
    """

    setUp = lv_setup

    def test_sim(self):
        pass

    def test_pf(self):
        pass

    def test_loglik(self):
        pass

    def test_state_sample(self):
        # un-self setUp members
        key = self.key
        theta = self.theta
        x_init = self.x_init
        model_args = self.model_args
        n_res = model_args["n_res"]
        n_obs = self.n_obs
        n_particles = self.n_particles
        model = self.Model(**model_args)
        # generate previous timepoint
        key, subkey = random.split(key)
        x_prev = jnp.block([[jnp.zeros((n_res-1, 2))],
                            [jnp.log(jnp.array([5., 3.]))]])
        x_prev = x_prev + random.normal(subkey, x_prev.shape)
        # simulate state using for-loop
        x_state1 = model.state_sample_for(key, x_prev, theta)
        # simulate state using lax.scan
        x_state2 = model.state_sample(key, x_prev, theta)
        self.assertAlmostEqual(utils.rel_err(x_state1, x_state2), 0.0)

    def test_state_lpdf(self):
        # un-self setUp members
        key = self.key
        theta = self.theta
        x_init = self.x_init
        model_args = self.model_args
        n_res = model_args["n_res"]
        n_obs = self.n_obs
        n_particles = self.n_particles
        model = self.Model(**model_args)
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
        self.assertAlmostEqual(utils.rel_err(lp1, lp2), 0.0)
