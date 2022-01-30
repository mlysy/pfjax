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
import pfjax as pf
import utils
from jax.config import config
config.update("jax_enable_x64", True)


class TestFact(unittest.TestCase):

    setUp = utils.fact_setup

    def test_tri_fact(self):
        """
        Check if p(W, X, Y) = p(W) p(X|W) p(Y|X, W).
        """
        # joint distribution using factorization
        lpdf1 = jsp.stats.multivariate_normal.logpdf(self.W, self.mu_W, self.Sigma_W)
        lpdf1 = lpdf1 + jsp.stats.multivariate_normal.logpdf(self.X, self.W + self.mu_XW, self.Sigma_XW)
        lpdf1 = lpdf1 + \
            jsp.stats.multivariate_normal.logpdf(self.Y, jnp.matmul(self.A, self.X), self.Omega)

        # joint distribution using single mvn
        lpdf2 = jsp.stats.multivariate_normal.logpdf(jnp.block([self.W, self.X, self.Y]), self.mu, self.Sigma)

        self.assertAlmostEqual(utils.rel_err(lpdf1, lpdf2), 0.0)

    def test_double_fact(self):
        """
        Check if p(W, Y) = p(Y) P(W|Y).
        """
        # joint distribution using factorization
        mu_WY, Sigma_WY = pf.sde.mvn_bridge_pars(self.mu_W, self.Sigma_W, self.mu_XW, 
                                             self.Sigma_XW, self.Y, self.A, self.Omega)
        lpdf1 = jsp.stats.multivariate_normal.logpdf(self.Y, self.mu_Y, self.Sigma_Y)
        lpdf1 = lpdf1 + jsp.stats.multivariate_normal.logpdf(self.W, mu_WY, Sigma_WY)

        # joint distribution using single mvn
        ind = jnp.concatenate([jnp.arange(self.n_lat), 2*self.n_lat + jnp.arange(self.n_obs)])
        lpdf2 = jsp.stats.multivariate_normal.logpdf(jnp.block([self.W, self.Y]), self.mu[ind], 
                                                     self.Sigma[jnp.ix_(ind, ind)])

        self.assertAlmostEqual(utils.rel_err(lpdf1, lpdf2), 0.0)
        
if __name__ == '__main__':
    unittest.main()
