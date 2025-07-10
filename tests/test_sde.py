"""
Unit tests for `pfjax.sde` module.

Things to test:

- [x] SDE base class works as expected, i.e., switches between `euler_{sim/lpdf}_diag()` and `euler_{sim/lpdf}_var()` at instantiation.

- [x] `jit` + `grad` return without errors.

- [x] JAX constructs (e.g., `vmap`, `xmap`, `lax.scan`, etc.) give the same result as for-loops, etc.

Test code: from `pfjax/tests`:

```
python -m unittest -v test_sde
```
"""

import unittest

import utils


class TestInheritDiag(unittest.TestCase):
    """
    Check that inheritance from SDEModel with `diff_diag=True` works as expected.
    """

    setUp = utils.lv_setup

    test_sim = utils.test_simulate_models
    test_loglik = utils.test_loglik_full_models


class TestInheritDense(unittest.TestCase):
    """
    Check that inheritance from SDEModel with `diff_diag=False` works as expected.
    """

    setUp = utils.sv_setup

    test_sim = utils.test_simulate_models
    test_loglik = utils.test_loglik_full_models


class TestJitDiag(unittest.TestCase):
    """
    Check that jitted code with `diff_diag=True` works as expected.
    """

    setUp = utils.lv_setup

    test_sim = utils.test_simulate_jit
    test_loglik = utils.test_loglik_full_jit


class TestJitDense(unittest.TestCase):
    """
    Check that jitted code with `diff_diag=False` works as expected.
    """

    setUp = utils.sv_setup

    test_sim = utils.test_simulate_jit
    test_loglik = utils.test_loglik_full_jit


class TestFor(unittest.TestCase):
    """
    Check that vmap/lax.scan are the same as for-loop (`LotVolModel` only).
    """

    setUp = utils.lv_setup

    test_state_sample = utils.test_sde_state_sample_for
    test_state_lpdf = utils.test_sde_state_lpdf_for
    # test_bridge_step = utils.test_sde_bridge_step_for


if __name__ == "__main__":
    unittest.main()
