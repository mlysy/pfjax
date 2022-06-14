"""
Unit tests for the PGNET Model.

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
# from jax.config import config
# config.update("jax_enable_x64", True)


class TestFor(unittest.TestCase):
    """
    Test whether for-loop version of functions is identical to xmap/scan version.
    """

    setUp = utils.pg_setup

    test_state_sample = utils.test_for_sde_state_sample
    test_state_lpdf = utils.test_for_sde_state_lpdf


if __name__ == '__main__':
    unittest.main()
