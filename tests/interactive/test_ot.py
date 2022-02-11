# testing the jax-ott library

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import ott
from ott.geometry import pointcloud
from ott.core import sinkhorn
from ott.tools import transport


# --- check ott vs custom sinkhorn algorithm -----------------------------------

def Teps(a, f, c, eps):
    """
    Sinkhorn dual transformation.
    """
    return -eps * jsp.special.logsumexp(jnp.log(a) + (f - c)/eps)


def potentials(a, b, u, v, eps, n_iter):
    """
    Sinkhorn algorithm as described in Corenflos et al (2021).

    Returns f, g, P, and C.
    """
    n = a.size
    f = jnp.zeros((n,))
    g = jnp.zeros((n,))
    P = jnp.zeros((n, n))
    C = jnp.zeros((n, n))
    # C = jnp.outer(u, u) + jnp.outer(v, v) - 2.0 * jnp.outer(u, v)
    for i in range(n):
        for j in range(n):
            C = C.at[i, j].set(jnp.sum(jnp.square(u[i] - v[j])))

    for t in range(n_iter):
        for i in range(n):
            f = f.at[i].set(0.5 * (f[i] + Teps(b, g, C[i, :], eps)))
            g = g.at[i].set(0.5 * (g[i] + Teps(a, f, C[:, i], eps)))

    # P = jnp.array([f] * n).T + jnp.array([g] * n) - C
    # P = jnp.outer(a, b) * jnp.exp(P / eps)
    for i in range(n):
        for j in range(n):
            P = P.at[i, j].set(a[i] * b[j] *
                               jnp.exp((f[i] + g[j] - C[i, j])/eps))

    return f, g, P, C

# compare potentials to ott version


key = random.PRNGKey(0)

# simulate data
n = 5  # size of problem
eps = .1
key, *subkeys = random.split(key, 5)
a = random.dirichlet(subkeys[0], alpha=jnp.ones((n,)))
b = random.dirichlet(subkeys[1], alpha=jnp.ones((n,)))
# NOTE: u and v must be 2d arrays, with 1st dim the number of points
u = random.normal(subkeys[2], shape=(n, 1))
v = random.normal(subkeys[3], shape=(n, 1))

# using manual code
f1, g1, P1, C = potentials(a, b, u, v, eps=eps, n_iter=1000)

# check first order condition
jnp.array([jnp.array([Teps(b, g1, C[i, :], eps) for i in range(n)]),
           f1])
jnp.array([jnp.array([Teps(a, f1, C[:, i], eps) for i in range(n)]),
           g1])

# check margins of OT matrix
jnp.array([sum(P1), b])
jnp.array([sum(P1.T), a])

# using ott-jax
geom = pointcloud.PointCloud(u, v, epsilon=eps)
out = sinkhorn.sinkhorn(geom, a, b)
P = geom.transport_from_potentials(out.f, out.g)

# difference between methods
jnp.abs(P1 - P) / (jnp.abs(P) + .1)

# --- example from documentation -----------------------------------------------


def create_points(rng, n, m, d):
    rngs = jax.random.split(rng, 3)
    x = jax.random.normal(rngs[0], (n, d)) + 1
    y = jax.random.uniform(rngs[1], (m, d))
    a = jnp.ones((n,)) / n
    b = jnp.ones((m,)) / m
    return x, y, a, b


rng = jax.random.PRNGKey(0)
n, m, d = 12, 14, 2
x, y, a, b = create_points(rng, n=n, m=m, d=d)

geom = pointcloud.PointCloud(x, y, epsilon=1e-2)
out = sinkhorn.sinkhorn(geom, a, b)
P = geom.transport_from_potentials(out.f, out.g)
