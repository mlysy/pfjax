"""
Utility functions.
"""

import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu


def logw_to_prob(logw):
    r"""
    Calculate normalized probabilities from unnormalized log weights.

    Args:
        logw: Vector of `n_particles` unnormalized log-weights.

    Returns:
        Vector of `n_particles` normalized weights that sum to 1.
    """
    wgt = jnp.exp(logw - jnp.max(logw))
    prob = wgt / jnp.sum(wgt)
    return prob


def rm_keys(x, keys):
    r"""
    Remove specified keys from given dict.
    """
    return {k: x[k] for k in x.keys() if k not in keys}


# --- tree helper functions ----------------------------------------------------


def tree_array2d(tree, shape0=None):
    r"""
    Convert a PyTree into a 2D JAX array.

    Starts by converting each leaf array to a 2D JAX array with same leading dimension.  Then concatenates these arrays along `axis=1`.  Assumes the leading dimension of each leaf is the same.

    **Notes:**

    - This function returns a tuple containing a Callable, so can't be jitted directly.  Can however be called in jitted code so long as the output is a PyTree.

    Args:
        tree: A Pytree.
        shape0: Optional value of the leading dimension.  If `None` is deduced from `tree`.

    Returns:
        tuple:
        - **array2d** - A two dimensional JAX array.
        - **unravel_fn** - A Callable to reconstruct the original PyTree.
    """
    if shape0 is None:
        shape0 = jtu.tree_leaves(tree)[0].shape[0]  # leading dimension
    y, _unravel_fn = jfu.ravel_pytree(tree)
    y = jnp.reshape(y, (shape0, -1))

    def unravel_fn(array2d):
        return _unravel_fn(jnp.ravel(array2d))

    return y, unravel_fn


def tree_add(tree1, tree2):
    r"""
    Add two pytrees leafwise.
    """
    return jtu.tree_map(lambda x, y: x + y, tree1, tree2)
    # return jtu.tree_map(lambda x, y: x+y[0], tree1, (tree2,))


def tree_mean(tree, logw):
    r"""
    Weighted mean of each leaf of a pytree along leading dimension.
    """
    prob = logw_to_prob(logw)
    broad_mult = jtu.Partial(jax.vmap(jnp.multiply), prob)
    return jtu.tree_map(jtu.Partial(jnp.sum, axis=0), jtu.tree_map(broad_mult, tree))


def tree_subset(tree, index):
    """
    Subset the leading dimension of each leaf of a pytree by values in index.
    """
    return jtu.tree_map(lambda x: x[index, ...], tree)


def tree_zeros(tree):
    r"""
    Fill pytree with zeros.
    """
    return jtu.tree_map(lambda x: jnp.zeros_like(x), tree)


def tree_remove_last(tree):
    """
    Remove last element of each leaf of pytree.
    """
    return jtu.tree_map(lambda _x: _x[:-1], tree)


def tree_remove_first(tree):
    """
    Remove first element of each leaf of pytree.
    """
    return jtu.tree_map(lambda _x: _x[1:], tree)


def tree_keep_last(tree):
    """
    Keep only last element of each leaft of pytree.
    """
    return jtu.tree_map(lambda _x: _x[-1], tree)


def tree_append_first(tree, first):
    """
    Append `first` to start of each leaf of `tree` along 1st dimension.
    """
    return jtu.tree_map(lambda x0, _x: jnp.concatenate([x0[None], _x]), first, tree)


def tree_append_last(tree, last):
    """
    Append `last` to end of each leaf of `tree` along 1st dimension.
    """
    return jtu.tree_map(lambda xl, _x: jnp.concatenate([_x, xl[None]]), last, tree)


# def tree_leading_dim(tree):
#     """
#     Compute the leading dimension of `tree`.

#     If this isn't the same for each leaf, return jnp.nan.  A consequence though
#     is that return type is float instead of int.

#     Also, not sure what to do about gradients through this...
#     """
#     leading = jtu.tree_map(lambda x: jnp.atleast_1d(x).shape[0], tree)
#     leading_min = jtu.tree_reduce(lambda x, y: jnp.minimum(x, y), leading)
#     leading_max = jtu.tree_reduce(lambda x, y: jnp.maximum(x, y), leading)
#     return jnp.where(leading_min < leading_max, jnp.nan, leading_min)
