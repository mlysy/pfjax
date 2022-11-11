"""
Unit tests for the factorizations of pdfs used in the bridge proposal.
In particular, suppose we are given

```
        W ~ N(\mu_W, \Sigma_W)
    X | W ~ N(W + \mu_XW, \Sigma_XW)
Y | X, W ~ N(AX, \Omega)
```

We are interested in factoring the pdf p(W, X, Y)= p(W) p(X|W) p(Y|X, W) 
and P(W, Y) = P(Y) P(W|Y).
"""
import unittest
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
from pfjax.mvn_bridge import *
import utils
# from jax.config import config
# config.update("jax_enable_x64", True)


class TestFact(unittest.TestCase):

    def setUp(self):
        """
        Creates the variables used in the tests for factorization.
        """
        key = random.PRNGKey(0)
        self.n_lat = 3  # number of dimensions of W and X
        self.n_obs = 2  # number of dimensions of Y
        # W and its mean and variance
        key, *subkeys = random.split(key, num=4)
        self.mu_W = random.normal(subkeys[0], (self.n_lat,))
        self.Sigma_W = utils.var_sim(subkeys[1], self.n_lat)
        self.W = random.normal(subkeys[2], (self.n_lat,))
        # X and its conditional mean and variance given W
        key, *subkeys = random.split(key, num=4)
        self.mu_XW = random.normal(subkeys[0], (self.n_lat,))
        self.Sigma_XW = utils.var_sim(subkeys[1], self.n_lat)
        self.X = random.normal(subkeys[2], (self.n_lat,))
        # Y and its conditional mean and variance given X
        key, *subkeys = random.split(key, num=4)
        self.A = random.normal(subkeys[0], (self.n_obs, self.n_lat))
        self.Omega = utils.var_sim(subkeys[1], self.n_obs)
        self.Y = random.normal(subkeys[2], (self.n_obs,))
        # joint distribution using single mvn
        self.mu_Y = jnp.matmul(self.A, self.mu_W + self.mu_XW)
        self.Sigma_Y = jnp.linalg.multi_dot(
            [self.A, self.Sigma_W + self.Sigma_XW, self.A.T]) + self.Omega
        AS_W = jnp.matmul(self.A, self.Sigma_W)
        AS_XW = jnp.matmul(self.A, self.Sigma_W + self.Sigma_XW)
        self.mu = jnp.block([self.mu_W, self.mu_W + self.mu_XW, self.mu_Y])
        self.Sigma = jnp.block([
            [self.Sigma_W, self.Sigma_W, AS_W.T],
            [self.Sigma_W, self.Sigma_W + self.Sigma_XW, AS_XW.T],
            [AS_W, AS_XW, self.Sigma_Y]
        ])

    def test_tri_fact(self):
        """
        Check if p(W, X, Y) = p(W) p(X|W) p(Y|X, W).
        """
        # joint distribution using factorization
        lpdf1 = jsp.stats.multivariate_normal.logpdf(
            self.W, self.mu_W, self.Sigma_W)
        lpdf1 = lpdf1 + \
            jsp.stats.multivariate_normal.logpdf(
                self.X, self.W + self.mu_XW, self.Sigma_XW)
        lpdf1 = lpdf1 + \
            jsp.stats.multivariate_normal.logpdf(
                self.Y, jnp.matmul(self.A, self.X), self.Omega)
        # joint distribution using single mvn
        lpdf2 = jsp.stats.multivariate_normal.logpdf(
            jnp.block([self.W, self.X, self.Y]), self.mu, self.Sigma)
        self.assertAlmostEqual(utils.rel_err(lpdf1, lpdf2), 0.0)

    def test_double_fact(self):
        """
        Check if p(W, Y) = p(Y) P(W|Y).
        """
        # joint distribution using factorization
        mu_Y, AS_W, Sigma_Y = mvn_bridge_pars(
            self.mu_W, self.Sigma_W, self.mu_XW,
            self.Sigma_XW, self.A, self.Omega
        )
        mu_WY, Sigma_WY = mvn_bridge_mv(self.mu_W, self.Sigma_W, mu_Y,
                                        AS_W, Sigma_Y, self.Y)
        lpdf1 = jsp.stats.multivariate_normal.logpdf(
            self.Y, mu_Y, Sigma_Y)
        lpdf1 = lpdf1 + \
            jsp.stats.multivariate_normal.logpdf(self.W, mu_WY, Sigma_WY)
        # joint distribution using single mvn
        ind = jnp.concatenate(
            [jnp.arange(self.n_lat), 2*self.n_lat + jnp.arange(self.n_obs)])
        lpdf2 = jsp.stats.multivariate_normal.logpdf(
            jnp.block([self.W, self.Y]), self.mu[ind],
            self.Sigma[jnp.ix_(ind, ind)]
        )
        self.assertAlmostEqual(utils.rel_err(lpdf1, lpdf2), 0.0)


if __name__ == '__main__':
    unittest.main()
