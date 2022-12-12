
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import pfjax as pf
import utils

class TestStoich(unittest.TestCase):
    setUp = utils.gnet_setup

    test_drift = utils.test_drift
    test_drift_mask = utils.test_drift_mask

if __name__ == '__main__':
    unittest.main()