"""
Utility functions.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu


def continuous_cdf (xs, weights, u):
    """
    Return one sample from a continuous approximation of the ECDF of x.
    
    Should bbe trivial to extend to multiple samples if needed by: 
        list(map(lambda x: np.argmax(w_cdf > x), u))
    """
    n = len(weights)
    
    pi = jnp.zeros(n + 1)
    pi = pi.at[0].set(weights[0] / 2)
    pi = pi.at[n].set(weights[-1] / 2)
    pi = pi.at[1:n].set((weights[:-1] + weights[1:]) / 2)
    
    w_cdf = jnp.cumsum(pi)
    r = jnp.argmax(w_cdf > u)
    u_new = (u - w_cdf[r-1]) / pi[r]
    
    # select region: 
    new_x = jax.lax.cond(
        r == 0,
        lambda _: xs[0],
        lambda _: jax.lax.cond(
            r == n,
            lambda _: xs[-1],
            lambda _: (xs[r] - xs[r-1]) * u_new + xs[r-1],
            r
        ),
        r
    )
    return new_x


def sort_marginal (x, weights):
    r""" 
    Sort X and return corresponding w
    """
    sorted_x, sorted_w = jax.lax.sort_key_val(x, weights)
    return {"x": sorted_x, "w": sorted_w}


def weighted_corr (X, weights):
    r"""
    Return weighted correlation matrix
    """
    corr_mat = jnp.cov(X, aweights = weights)
    stddevs = jnp.sqrt(jnp.diag(corr_mat))
    corr_mat = corr_mat / stddevs[:, None] / stddevs[None, :]
    return corr_mat


def marginal_cdf (x, data, weights):
    r"""
    Return quantile of x given data and weights
    """
    return sum(jnp.where(data < x, x = weights, y=0))


def lwgt_to_prob(logw):
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


def tree_add(tree1, tree2):
    r"""
    Add two pytrees leafwise.
    """
    return jtu.tree_map(lambda x, y: x+y, tree1, tree2)
    # return jtu.tree_map(lambda x, y: x+y[0], tree1, (tree2,))


def tree_mean(tree, logw):
    r"""
    Weighted mean of each leaf of a pytree along leading dimension.
    """
    prob = lwgt_to_prob(logw)
    broad_mult = jtu.Partial(jax.vmap(jnp.multiply), prob)
    return jtu.tree_map(
        jtu.Partial(jnp.sum, axis=0),
        jtu.tree_map(broad_mult, tree)
    )


def tree_shuffle(tree, index):
    """
    Shuffle the leading dimension of each leaf of a pytree by values in index.
    """
    return jtu.tree_map(lambda x: x[index, ...], tree)


def tree_zeros(tree):
    r"""
    Fill pytree with zeros.
    """
    return jtu.tree_map(lambda x: jnp.zeros_like(x), tree)


def tree_remove_last(x):
    """
    Remove last element of each leaf of pytree.
    """
    return jtu.tree_map(lambda _x: _x[:-1], x)


def tree_remove_first(x):
    """
    Remove first element of each leaf of pytree.
    """
    return jtu.tree_map(lambda _x: _x[1:], x)


def tree_keep_last(x):
    """
    Keep only last element of each leaft of pytree.
    """
    return jtu.tree_map(lambda _x: _x[-1], x)


def tree_append_first(x, first):
    """
    Append `first` to start of each leaf of `x` along 1st dimension.
    """
    return jtu.tree_map(lambda x0, _x: jnp.concatenate([x0[None], _x]), first, x)


def tree_append_last(x, last):
    """
    Append `last` to end of each leaf of `x` along 1st dimension.
    """
    return jtu.tree_map(lambda xl, _x: jnp.concatenate([_x, xl[None]]), last, x)
