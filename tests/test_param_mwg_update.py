"""
Unit tests for `pfjax.mcmc.param_mwg_update()`.

Things to test:

- [x] `vmap`, `xmap`, `scan`, etc. give the same result as with for-loops.

- [x] `jit` + `grad` return without errors.

Test code: from `pfjax/tests`:

```
python -m unittest -v test_param_mwg_update
```
"""

import unittest
import utils


class TestBMModel(unittest.TestCase):

    setUp = utils.bm_setup

    test_for = utils.test_param_mwg_update_for
    test_jit = utils.test_param_mwg_update_jit


if __name__ == '__main__':
    unittest.main()
