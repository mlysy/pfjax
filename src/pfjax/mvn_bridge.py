r"""
Multivariate normal bridge proposals.

Suppose we have the multivariate normal model

::

           W ~ N(mu_W, Sigma_W)
       X | W ~ N(W + mu_XW, Sigma_XW)
    Y | X, W ~ N(AX, Omega).

We are interested in calculating the mean and variance of `p(W|Y)`.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp


def mvn_bridge_pars(mu_W, Sigma_W, mu_XW, Sigma_XW, A, Omega):
    r"""
    Calculate the unconditional mean of Y, the variance of Y and the covariance between W and Y.

    Args:
        mu_W: Mean of W.
        Sigma_W: Variance of W.
        mu_XW: Mean fo X|W.
        Sigma_XW: Variance of X|W.
        A: Matrix to obtain mean of Y given X,W.
        Omega: Variance of Y|X,W.

    Returns:
        Tuple:

        - **mu_Y** - Unconditional mean of Y.
        - **AS_W** - Covariance of W, Y.
        - **Sigma_Y** - Unconditional variance of Y.

    """
    mu_Y = jnp.matmul(A, mu_W + mu_XW)
    AS_W = jnp.matmul(A, Sigma_W)
    Sigma_Y = jnp.linalg.multi_dot([A, Sigma_W + Sigma_XW, A.T]) + Omega
    return mu_Y, AS_W, Sigma_Y


def mvn_bridge_mv(mu_W, Sigma_W, mu_Y, AS_W, Sigma_Y, Y):
    r"""
    Calculate the mean and variance of `p(W|Y)`.

    Args:
        mu_W: Mean of W.
        Sigma_W: Variance of W.
        mu_Y: Unconditional mean of Y.
        AS_W: Covariance of Y, W.
        Sigma_Y: Unconditional variance of Y.
        Y: Observed Y.

    Returns:
        Tuple:

        - **mu_WY** - Mean of W|Y.
        - **Sigma_WY** - Variance of W|Y.

    """
    # solve both linear systems simultaneously
    # sol = jnp.matmul(AS_W.T, jnp.linalg.solve(
    #     Sigma_Y, jnp.hstack([jnp.array([Y-mu_Y]).T, AS_W])))

    Sigma_chol = jsp.linalg.cho_factor(Sigma_Y, True)
    sol = jnp.matmul(AS_W.T, jsp.linalg.cho_solve(
        Sigma_chol, jnp.hstack([jnp.array([Y-mu_Y]).T, AS_W])))
    return mu_W + jnp.squeeze(sol[:, 0]), Sigma_W - sol[:, 1:]
