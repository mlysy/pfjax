"""
Unit tests for `pfjax.particle_filter()`.

Things to test:

- [x] `vmap`, `xmap`, `scan`, etc. give the same result as with for-loops.

- [x] `jit` + `grad` return without errors.

Test code: from `pfjax/tests`:

```
python -m unittest -v test_particle_filter
```
"""

import unittest
import utils


class TestBMModel(unittest.TestCase):

    setUp = utils.bm_setup

    test_for = utils.test_particle_filter_for
    test_deriv = utils.test_particle_filter_deriv


if __name__ == '__main__':
    unittest.main()
