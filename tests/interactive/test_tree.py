import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu

# --- test repeats -------------------------------------------------------------

theta = (
    jnp.array(1.0),
    (jnp.arange(2) * 1.0, jnp.array([[jnp.arange(2) + 1.0], [jnp.arange(2) * 2.0]])),
)

jax.hessian(lambda x: 0.0)(theta)

# --- test sum -----------------------------------------------------------------


def tree_sum(tree1, tree2):
    """
    Add two pytrees.
    """
    return jtu.tree_map(lambda x, y: x + y, tree1, tree2)
    # return jtu.tree_map(lambda x, y: x+y[0], tree1, (tree2,))


# --- test with simple arrays --------------------------------------------------

x = jnp.array(1.0)
y = jnp.array(2.0)

tree_sum(x, y)

# --- test with dict -----------------------------------------------------------

x = {"a": jnp.array(1.0), "b": jnp.linspace(1.0, 2.0, 5)}
y = {"a": jnp.array(2.0), "b": jnp.linspace(2.0, 7.0, 5)}

tree_sum(x, y)

# --- test with tuple ----------------------------------------------------------

x = (jnp.array([1.0, 2.0]), jnp.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]))
y = (jnp.array([5.0, 3.0]), jnp.array([[0.0, -1.0], [-3.0, 2.0]]))

tree_sum(x, y)


# --- test tree flatten/unflatten ----------------------------------------------

add_one = jtu.Partial(jnp.add, jnp.array([1.0, 3.0]))

jtu.tree_map(add_one, x)

jtu.tree_map(jtu.Partial(jnp.sum, axis=0), jtu.tree_map(add_one, x))

# --- flatten all but one dimension --------------------------------------------


def reshape_2d(x):
    """
    Reshape an array to have two-dimensions.

    - If it has zero or one dimension, turns into a column matrix.
    - If it has more than two dimensions, leading dimension stays the same.

    Returns:
       Reshaped array.
    """
    shape = x.shape
    if len(shape) < 2:
        y = jnp.atleast_2d(x)
    else:
        y = jnp.reshape(x, newshape=(shape[0], -1))
    # return y, shape
    return y


def tree_array2d(x, shape0=None):
    """
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

    def unravel_fn(array2d):
        return _unravel_fn(jnp.ravel(array2d))

    return y, unravel_fn


n_particles = 5
x = {
    "a": jnp.tile(jnp.array([1.0, 2.0]), reps=(n_particles, 1)),
    "b": jnp.tile(
        jnp.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), reps=(n_particles, 1, 1)
    ),
}


def foo(x, shape0=None):
    y, unravel_fn = tree_array2d(x, shape0)
    return unravel_fn(y)


foo_jit = jax.jit(foo, static_argnames="shape0")


jtu.tree_map(lambda a, b: a == b, x, foo_jit(x))

jtu.tree_map(lambda a, b: a == b, x, foo_jit(x, shape0=n_particles))
