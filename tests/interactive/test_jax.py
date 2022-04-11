import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
from functools import partial
import pfjax as pf

# --- empty dictionaries -------------------------------------------------------

x = {"a": jnp.array(5.), "b": jnp.array([1., 2.])}
{k: x[k] for k in x.keys() if k in "c"}

jax.vmap(lambda x: {})(jnp.arange(10))

# --- none as static argument? -------------------------------------------------


def foo(x, fun):
    if fun is None:
        return jnp.array(0.)
    else:
        return fun(jnp.array(x))


def bar(x):
    return x + jnp.array(7.)


jfoo = jax.jit(partial(foo, fun=bar))

# --- see how append works -----------------------------------------------------


x = []
y = jnp.array([[0.], [1.]])
x.append(y)
x.append(y)
jnp.array(x)

# --- reshape ------------------------------------------------------------------

n_devices = 4
n_particles_per_device = 3

x = jnp.array(jax.random.normal(
    key, (n_devices, n_particles_per_device, 5, 2)))

y = x.reshape((-1,) + x.shape[2:])

x.shape
y.shape

z = y.reshape((n_devices, n_particles_per_device) + y.shape[1:])
