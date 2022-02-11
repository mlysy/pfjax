import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import pfjax as pf

# see how append works

x = []
y = jnp.array([[0.], [1.]])
x.append(y)
x.append(y)
jnp.array(x)
