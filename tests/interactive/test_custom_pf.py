import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import particle_filter as pf
import bm_model as bm

key = random.PRNGKey(0)
# parameter values
mu = 5
sigma = 1
tau = .1
theta = jnp.array([mu, sigma, tau])
# data specification
dt = .1
n_obs = 5
x_init = jnp.array([0.])
bm_model = bm.BMModel(dt=dt)
# simulate without for-loop
y_meas, x_state = pf.meas_sim(bm_model, n_obs, x_init, theta, key)
# particle filter specification
n_particles = 7
key, subkey = random.split(key)
# pf without for-loop
pf_out = pf.particle_filter(
    bm_model, y_meas, theta, n_particles, subkey)
