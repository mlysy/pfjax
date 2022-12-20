"""
Utility functions.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
import jax.flatten_util as jfu



def diameter(x, y):
    """
    Helper function for scaling of particles used in optimal transport resampling
    """
    diameter_x = jnp.max(jnp.std(x, 1), -1)
    diameter_y = jnp.max(jnp.std(y, 1), -1)
    res = jnp.maximum(diameter_x, diameter_y)
    return jnp.where(res == 0., 1., res)


def scale_x (x):
    """
    Scale X as done in http://proceedings.mlr.press/v139/corenflos21a/corenflos21a.pdf.
    
    This replicated their code here: 
    https://github.com/JTT94/filterflow/blob/master/filterflow/resampling/differentiable/regularized_transport/plan.py#L67
    """
    centered_x = x - jnp.mean(x, axis=1, keepdims=True)
    diameter_value = diameter(x, x)
    scale = jnp.reshape(diameter_value, [-1, 1, 1]) * jnp.sqrt(x.shape[1])
    scaled_x = centered_x / scale
    return scaled_x.reshape(x.shape)


def ess (normalized_weights):
    """ 
    Calculate ffective sample size from normalized weights
    """
    return 1 / sum(normalized_weights ** 2)


def continuous_cdf (xs, pi, u):
    """
    Return a sample from a continuous approximation of the ECDF of x.

    Args: 
        - xs: Sorted marginals
        - pi: interpolated weights of each x. Should be of length: len(xs) + 1
        - u: U(0,1)
    
    Returns: 
        Sample
    """
    n = len(xs)    

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


def interpolate_weights (weights):
    """ Interpolate weights as in Malik&Pitt """
    n = len(weights) 
    pi = jnp.zeros(n + 1)
    pi = pi.at[0].set(weights[0] / 2)
    pi = pi.at[n].set(weights[-1] / 2)
    pi = pi.at[1:n].set((weights[:-1] + weights[1:]) / 2)
    return pi


def argsort_marginal (x, w):
    """ sort (x,w) based on x and return the indices of the sort """
    sort_indices = jnp.argsort(x)
    return {"x": jnp.take(x, sort_indices),
            "w": jnp.take(w, sort_indices),
            "unsorted_x": x,
            "indices": sort_indices}


def quantile_func (x, sorted_samples, cumulative_weights):
    """
    x: vector of points to evaluate cdf
    cdf: {"x": sorted samples
          "w": cumulative sum of weights}
    """
    cdf_fn = jax.vmap(lambda y: cumulative_weights[jnp.argmax(sorted_samples == y)])
    return cdf_fn(x)


def weighted_corr (X, weights):
    r"""
    Return weighted correlation matrix
    """
    corr_mat = jnp.cov(X, aweights = weights)
    stddevs = jnp.sqrt(jnp.diag(corr_mat))
    corr_mat = corr_mat / stddevs[:, None] / stddevs[None, :]
    return corr_mat


# def marginal_cdf (x, data, weights):
#     r"""
#     Return quantile of x given data and weights
#     """
#     return sum(jnp.where(data <= x, x = weights, y=0))


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

def tree_array2d(x, shape0=None):
    r"""
    Convert a PyTree into a 2D JAX array.

    Starts by converting each leaf array to a 2D JAX array with same leading dimension.  Then concatenates these arrays along `axis=1`.  Assumes the leading dimension of each leaf is the same.

    **Notes:**

    - This function returns a tuple containing a Callable, so can't be jitted directly.  Can however be called in jitted code so long as the output is a PyTree.

    Args:
        x: A Pytree.
        shape0: Optional value of the leading dimension.  If `None` is deduced from `x`.

    Returns:
        tuple:
        - **array2d** - A two dimensional JAX array.
        - **unravel_fn** - A Callable to reconstruct the original PyTree.
    """
    if shape0 is None:
        shape0 = jtu.tree_leaves(x)[0].shape[0]  # leading dimension
    y, _unravel_fn = jfu.ravel_pytree(x)
    y = jnp.reshape(y, (shape0, -1))
    def unravel_fn(array2d): return _unravel_fn(jnp.ravel(array2d))
    return y, unravel_fn


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
    prob = logw_to_prob(logw)
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
