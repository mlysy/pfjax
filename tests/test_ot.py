"""
Optimal transport tests.
"""
import unittest
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import ott
from ott.geometry import pointcloud
from ott.solvers.linear import sinkhorn
from pfjax.test import sinkhorn_test
import utils
# from jax.config import config
# config.update("jax_enable_x64", True)


class TestOT(unittest.TestCase):

    def test_sinkhorn(self):
        key = random.PRNGKey(0)
        # simulate data
        n = 5  # size of problem
        eps = .1
        key, *subkeys = random.split(key, 5)
        a = random.dirichlet(subkeys[0], alpha=jnp.ones((n,)))
        b = random.dirichlet(subkeys[1], alpha=jnp.ones((n,)))
        # NOTE: u and v must be 2d arrays, with 1st dim the number of points
        u = random.normal(subkeys[2], shape=(n, 2))
        v = random.normal(subkeys[3], shape=(n, 2))
        # sinkhorn with ott-jax
        geom = pointcloud.PointCloud(u, v, epsilon=eps)
        out = sinkhorn.sinkhorn(geom, a, b)
        P1 = out.matrix
        tsp1 = geom.apply_transport_from_potentials(
            f=out.f, g=out.g, vec=v.T, axis=1).T
        # sinkhorn with custom code
        _, _, P2, _ = sinkhorn_test(a, b, u, v, eps=eps, n_iter=1000)
        tsp2 = jnp.matmul(P1, v)  # Note: this is calculated with P1
        self.assertLess(utils.rel_err(P1, P2), 0.01)
        self.assertLess(utils.rel_err(tsp1, tsp2), 0.01)


if __name__ == '__main__':
    unittest.main()
