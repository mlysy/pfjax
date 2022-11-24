# testing the jax-ott library

import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import ott
from ott.geometry import pointcloud
from ott.core import sinkhorn
from ott.tools import transport
import pfjax as pf
# import pfjax.experimental.particle_filter as pfex
from pfjax.particle_resamplers import resample_ot, resample_multinomial
from pfjax.models.lotvol_model import LotVolModel

key = random.PRNGKey(0)
# parameter values
alpha = 1.02
beta = 1.02
gamma = 4.
delta = 1.04
sigma_H = .1
sigma_L = .2
tau_H = .25
tau_L = .35
theta = jnp.array([alpha, beta, gamma, delta, sigma_H, sigma_L, tau_H, tau_L])
# data specification
dt = .09
n_res = 10
n_obs = 100
x_init = jnp.block([[jnp.zeros((n_res-1, 2))],
                    [jnp.log(jnp.array([5., 3.]))]])
# simulate with inherited class
lv_model = LotVolModel(dt=dt, n_res=n_res)
y_meas, x_state = pf.simulate(lv_model, key, n_obs, x_init, theta)

n_particles = 100


def particle_filter(theta, y_meas, key):
    return pf.particle_filter(
        model=lv_model,
        key=key,
        theta=theta,
        y_meas=y_meas,
        n_particles=n_particles,
        score=False,
        fisher=False,
        history=True,
        resampler=resample_multinomial
    )


pf_out = jax.jit(particle_filter)(theta=theta, y_meas=y_meas[0:2], key=key)

# simplified ot interface: n_iterations and epsilon provided as constants
epsilon = jnp.array(1.0)
n_iterations = 10


# unjitted
def resample_ot_simple(x_particles_prev, logw, key):
    return resample_ot(
        x_particles_prev=x_particles_prev,
        logw=logw,
        key=key,
        sinkhorn_kwargs={"min_iterations": n_iterations,
                         "max_iterations": n_iterations},
        pointcloud_kwargs={"epsilon": epsilon}
    )


%timeit resample_ot_simple(x_particles_prev=pf_out["x_particles"][0], logw=pf_out["logw"][0], key=key)

# jitted
%timeit jax.jit(resample_ot_simple)(x_particles_prev=pf_out["x_particles"][0], logw=pf_out["logw"][0], key=key)

# try jitting a different way
# first unjitted
partial(resample_ot,
        sinkhorn_kwargs={"min_iterations": n_iterations,
                         "max_iterations": n_iterations},
        pointcloud_kwargs={"epsilon": epsilon})(
            x_particles_prev=pf_out["x_particles"][0],
            logw=pf_out["logw"][0],
            key=key
)

# now jitted
jax.jit(partial(resample_ot,
                sinkhorn_kwargs={"min_iterations": n_iterations,
                                 "max_iterations": n_iterations},
                pointcloud_kwargs={"epsilon": epsilon}))(
    x_particles_prev=pf_out["x_particles"][0],
    logw=pf_out["logw"][0],
    key=key
)

jax.jit(partial(resample_ot,
                sinkhorn_kwargs={"min_iterations": n_iterations,
                                 "max_iterations": n_iterations}))(
    x_particles_prev=pf_out["x_particles"][0],
    logw=pf_out["logw"][0],
    key=key,
)
jax.jit(resample_ot)(
    x_particles_prev=pf_out["x_particles"][0],
    logw=pf_out["logw"][0],
    key=key,
    sinkhorn_kwargs={"min_iterations": n_iterations,
                     "max_iterations": n_iterations}
    # pointcloud_kwargs={"epsilon": epsilon}
)

# unjitted


# jitted
@jax.jit
def resample_jit(x_particles_prev, logw, key):
    return resample_ot(
        x_particles_prev=x_particles_prev,
        logw=logw,
        key=key,
        sinkhorn_kwargs={"min_iterations": n_iterations,
                         "max_iterations": n_iterations}
        # pointcloud_kwargs={"epsilon": epsilon}
    )


resample_jit(
    x_particles_prev=pf_out["x_particles"][0],
    logw=pf_out["logw"][0],
    key=key
)

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
    # P = jnp.zeros((n, n))
    # C = jnp.zeros((n, n))

    # for i in range(n):
    #     for j in range(n):
    #         C = C.at[i, j].set(jnp.sum(jnp.square(u[i] - v[j])))
    C = jax.vmap(
        jax.vmap(lambda _u, _v: jnp.sum(jnp.square(_u - _v)),
                 in_axes=(None, 0)),
        in_axes=(0, None)
    )(u, v)
    CT = C.T

    # for t in range(n_iter):
    #     for i in range(n):
    #         f = f.at[i].set(0.5 * (f[i] + Teps(b, g, C[i, :], eps)))
    #         g = g.at[i].set(0.5 * (g[i] + Teps(a, f, C[:, i], eps)))

    def update(fg, t):
        f, g = fg
        f = 0.5 * (f + jax.vmap(lambda c: Teps(b, g, c, eps))(C))
        g = 0.5 * (g + jax.vmap(lambda c: Teps(a, f, c, eps))(CT))
        return (f, g), None

    fg, _ = jax.lax.scan(update, (f, g), jnp.arange(n_iter))
    f, g = fg

    # for i in range(n):
    #     for j in range(n):
    #         P = P.at[i, j].set(a[i] * b[j] *
    #                            jnp.exp((f[i] + g[j] - C[i, j])/eps))

    P = jax.vmap(
        jax.vmap(
            lambda _a, _b, _f, _g, _c: _a*_b * jnp.exp((_f + _g - _c)/eps),
            in_axes=(None, 0, None, 0, 0)
        ),
        in_axes=(0, None, 0, None, 0)
    )(a, b, f, g, C)
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
u = random.normal(subkeys[2], shape=(n, 2))
v = random.normal(subkeys[3], shape=(n, 2))

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
P2 = transport.solve(u, v, a=a, b=b, epsilon=eps, jit=False).matrix

# difference between methods
jnp.abs(P1 - P) / (jnp.abs(P) + .1)
jnp.abs(P2 - P) / (jnp.abs(P) + .1)

# now try applying transport to a vector
tr = geom.apply_transport_from_potentials(
    f=out.f, g=out.g, vec=v.T, axis=1).T
tr1 = jnp.matmul(P, v)

jnp.abs(tr1 - tr) / (jnp.abs(tr) + .1)

# --- test particle_resample_ot ------------------------------------------------

pf.particle_filter(lv_model, key,
                   y_meas, theta, n_particles,
                   particle_sampler=pf.particle_resample_ot)

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
