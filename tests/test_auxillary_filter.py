"""
Unit tests for the auxillary particle filter.

Things to test:

- [ ] `jit` compiles without errors.

- [ ] `vmap`, `xmap`, `scan`, etc. give the same result as with for-loops.

- [ ] With and without history gives the same result.
```
"""

import unittest
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

    def test_loglik(self):
        pass


if __name__ == '__main__':
    unittest.main()
