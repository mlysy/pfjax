import numpy as np
import jax.numpy as jnp


def rel_err(X1, X2):
    """
    Relative error between two JAX arrays.

    Adds 0.1 to the denominator to avoid nan's when its equal to zero.
    """
    return jnp.max(jnp.abs((X1.ravel() - X2.ravel())/(0.1 + X1.ravel())))
