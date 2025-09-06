"""
Optimal transport tests.
"""

import unittest
import utils


class TestOT(unittest.TestCase):

    setUp = utils.ot_setup

    test_sinkhorn = utils.test_resample_ot_sinkhorn
    test_jit = utils.test_resample_ot_jit


if __name__ == '__main__':
    unittest.main()
