import jax
import jax.numpy as jnp
import jax.tree_util as jtu


def tree_sum(tree1, tree2):
    """
    Add two pytrees.
    """
    return jtu.tree_map(lambda x, y: x+y, tree1, tree2)
    # return jtu.tree_map(lambda x, y: x+y[0], tree1, (tree2,))


# --- test with simple arrays --------------------------------------------------

x = jnp.array(1.)
y = jnp.array(2.)

tree_sum(x, y)

# --- test with dict -----------------------------------------------------------

x = {"a": jnp.array(1.), "b": jnp.linspace(1., 2., 5)}
y = {"a": jnp.array(2.), "b": jnp.linspace(2., 7., 5)}

tree_sum(x, y)

# --- test with tuple ----------------------------------------------------------

x = (jnp.array([1., 2.]), jnp.array([[1., 2., 3.], [3., 4., 5.]]))
y = (jnp.array([5., 3.]), jnp.array([[0., -1.], [-3., 2.]]))

tree_sum(x, y)


# --- test tree flatten/unflatten ----------------------------------------------

add_one = jtu.Partial(jnp.add, jnp.array([1., 3.]))

jtu.tree_map(add_one, x)

jtu.tree_map(jtu.Partial(jnp.sum, axis=0), jtu.tree_map(add_one, x))
