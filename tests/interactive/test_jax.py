from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy as jsp
import jax.tree_util as jtu
import numpy as np
import pfjax as pf

# --- leading dimensions -------------------------------------------------------

x = (1.0 * jnp.arange(2), (1.0 * jnp.arange(3), jnp.array(3.0)))
y = jtu.tree_map(lambda x: jnp.atleast_1d(x).shape[0], x)

jtu.tree_reduce(lambda x, y: jnp.minimum(x, y), y)
jtu.tree_reduce(lambda x, y: jnp.maximum(x, y), y)

pf.utils.tree_leading_dim(x)

x = (1.0 * jnp.arange(2), (1.0 * jnp.arange(2), 1.0 * jnp.arange(2)))
pf.utils.tree_leading_dim(x)


# --- vmap with key-value pairs ------------------------------------------------


def foo(a, b):
    return a**2 + b


grad_foo = jax.grad(foo)

grad_foo(a=2.0, b=3.0)

# --- broadcasting with vmap ---------------------------------------------------


prob = jnp.arange(5)
alpha = jnp.array([jnp.arange(3)] * 5)
beta = jnp.array([jnp.outer(jnp.arange(3), jnp.arange(3))] * 5)

jax.vmap(jnp.multiply)(prob, alpha)
jtu.tree_map(jax.vmap(jnp.multiply), (prob, prob), (alpha, beta))


# --- empty dictionaries -------------------------------------------------------


x = {"a": jnp.array(5.0), "b": jnp.array([1.0, 2.0])}
{k: x[k] for k in x.keys() if k in "c"}

jax.vmap(lambda x: {})(jnp.arange(10))


# --- none as static argument? -------------------------------------------------


def foo(x, fun):
    if fun is None:
        return jnp.array(0.0)
    else:
        return fun(jnp.array(x))


def bar(x):
    return x + jnp.array(7.0)


jfoo = jax.jit(partial(foo, fun=bar))


# --- see how append works -----------------------------------------------------


x = []
y = jnp.array([[0.0], [1.0]])
x.append(y)
x.append(y)
jnp.array(x)


# --- reshape ------------------------------------------------------------------

n_devices = 4
n_particles_per_device = 3

x = jnp.array(jax.random.normal(key, (n_devices, n_particles_per_device, 5, 2)))

y = x.reshape((-1,) + x.shape[2:])

x.shape
y.shape

z = y.reshape((n_devices, n_particles_per_device) + y.shape[1:])


# --- test replication ---------------------------------------------------------


def zero_pad(x, n):
    """Zero-pad an array along the leading dimension."""
    zeros = jnp.zeros((n - 1,) + x.shape)
    return jnp.concatenate([zeros, x[None]])


jax.jit(zero_pad, static_argnums=1)(x=jnp.ones((3, 2)), n=1)
