"""
Unit tests for SDE methods.

Things to test:

- [x] SDE base class works as expected, i.e., switches between `euler_{sim/lpdf}_diag()` and `euler_{sim/lpdf}_var()` at instantiation.

- [x] `jit` + `grad` return without errors.

- [x] JAX constructs (e.g., `vmap`, `xmap`, `lax.scan`, etc.) give the same result as for-loops, etc.

    In this case just need to check this for `state_sample` and `state_lpdf`, as inference method checks are conducted elsewhere.

"""

import unittest
# import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import pfjax as pf
import utils


class TestInherit(unittest.TestCase):
    """
    Check that inheritance from SDEModel works as expected.
    """

    setUp = utils.lv_setup

    test_sim = utils.test_models_simulate
    test_loglik = utils.test_models_loglik_full
    test_pf = utils.test_models_particle_filter


class TestJit(unittest.TestCase):
    """
    Check whether jit with and without grad gives the same result.
    """

    setUp = utils.lv_setup

    test_sim = utils.test_jit_simulate
    test_pf = utils.test_jit_particle_filter
    test_loglik = utils.test_jit_loglik_full


class TestFor(unittest.TestCase):
    """
    Test whether for-loop version of functions is identical to xmap/scan version.
    """

    setUp = utils.lv_setup

    test_state_sample = utils.test_for_sde_state_sample
    test_state_lpdf = utils.test_for_sde_state_lpdf

    # def test_state_sample(self):
    #     # un-self setUp members
    #     key = self.key
    #     theta = self.theta
    #     x_init = self.x_init
    #     model_args = self.model_args
    #     n_res = model_args["n_res"]
    #     n_obs = self.n_obs
    #     n_particles = self.n_particles
    #     model = self.Model(**model_args)
    #     # generate previous timepoint
    #     key, subkey = random.split(key)
    #     x_prev = jnp.block([[jnp.zeros((n_res-1, 2))],
    #                         [jnp.log(jnp.array([5., 3.]))]])
    #     x_prev = x_prev + random.normal(subkey, x_prev.shape)
    #     # simulate state using for-loop
    #     x_state1 = model.state_sample_for(key, x_prev, theta)
    #     # simulate state using lax.scan
    #     x_state2 = model.state_sample(key, x_prev, theta)
    #     self.assertAlmostEqual(utils.rel_err(x_state1, x_state2), 0.0)

    # def test_state_lpdf(self):
    #     # un-self setUp members
    #     key = self.key
    #     theta = self.theta
    #     x_init = self.x_init
    #     model_args = self.model_args
    #     n_res = model_args["n_res"]
    #     n_obs = self.n_obs
    #     n_particles = self.n_particles
    #     model = self.Model(**model_args)
    #     # generate previous timepoint
    #     key, subkey = random.split(key)
    #     x_prev = jnp.block([[jnp.zeros((n_res-1, 2))],
    #                         [jnp.log(jnp.array([5., 3.]))]])
    #     x_prev = x_prev + random.normal(subkey, x_prev.shape)
    #     # simulate state using lax.scan
    #     x_curr = model.state_sample(key, x_prev, theta)
    #     # lpdf using for
    #     lp1 = model.state_lpdf_for(x_curr, x_prev, theta)
    #     lp2 = model.state_lpdf(x_curr, x_prev, theta)
    #     self.assertAlmostEqual(utils.rel_err(lp1, lp2), 0.0)

    def test_bridge_prop(self):
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
        y_curr = jnp.exp(x_prev[-1]) + \
            theta[6:8] * random.normal(subkey, (x_prev.shape[1],))

        # bridge proposal using lax.scan
        x_curr1, logw1 = model.bridge_prop(
            key=key,
            x_prev=x_prev,
            y_curr=y_curr,
            theta=theta,
            Y=jnp.log(y_curr),
            A=jnp.eye(2),
            Omega=jnp.eye(2)
        )
        # bridge proposal using for
        x_curr2, logw2 = model.bridge_prop_for(
            key=key,
            x_prev=x_prev,
            y_curr=y_curr,
            theta=theta,
            Y=jnp.log(y_curr),
            A=jnp.eye(2),
            Omega=jnp.eye(2)
        )
        self.assertAlmostEqual(utils.rel_err(x_curr1, x_curr2), 0.0)
        self.assertAlmostEqual(utils.rel_err(logw1, logw2), 0.0)


if __name__ == '__main__':
    unittest.main()
