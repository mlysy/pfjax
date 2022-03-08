"""
Unit tests for MCMC methods.

Things to test:

- [x] JAX constructs (e.g., `vmap`, `xmap`, `lax.scan`, etc.) give the same result as for-loops, etc.

- [x] `jit` + `grad` return without errors.
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

    test_mwg = utils.test_for_mwg


class TestJit(unittest.TestCase):
    """
    Check whether jit with and without grad gives the same result.
    """

    setUp = utils.bm_setup

    test_mwg = utils.test_jit_mwg
