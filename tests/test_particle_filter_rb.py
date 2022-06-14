"""
Unit tests for `pfjax.particle_filter_rb()`.

Things to test:

- [x] `vmap`, `xmap`, `scan`, etc. give the same result as with for-loops.

- [x] `jit` + `grad` return without errors.

Test code: from `pfjax/tests`:

```
python -m unittest -v test_particle_filter_rb
```
"""

import unittest
import utils


class TestBMModel(unittest.TestCase):

    setUp = utils.bm_setup

    test_for = utils.test_particle_filter_rb_for
    test_hist = utils.test_particle_filter_rb_history
    test_deriv = utils.test_particle_filter_rb_deriv


if __name__ == '__main__':
    unittest.main()
