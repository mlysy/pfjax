import jax
import jax.numpy as jnp


def rel_err(X1, X2):
    """
    Relative error between two JAX arrays.

    Adds 0.1 to the denominator to avoid nan's when its equal to zero.
    """
    x1 = jnp.atleast_1d(X1).ravel() * 1.0
    x2 = jnp.atleast_1d(X2).ravel() * 1.0
    return jnp.max(jnp.abs(x1 - x2) / (0.1 + jnp.abs(x1)))


@jax.jit
def rel_err_tree(X1, X2):
    def _rel_err(x1, x2):
        err = rel_err(x1, x2)
        assert err < 1e-5

    jax.tree.map(_rel_err, X1, X2)


def test_tree():
    X1 = (
        jnp.arange(3),
        {"x": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "y": jnp.array(5.0)},
    )
    X2 = (
        jnp.arange(3),
        {"x": jnp.array([[1.0, 2.0], [3.0, 2.0]]), "y": jnp.array(5.0)},
    )
    rel_err_tree(X1, X2)
