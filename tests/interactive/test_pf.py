import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
from functools import partial
import pfjax as pf

key = random.PRNGKey(0)
# parameter values
mu = 5
sigma = 1
tau = .1
theta = jnp.array([mu, sigma, tau])
# data specification
dt = .1
n_obs = 5
x_init = jnp.array(0.)
bm_model = pf.BMModel(dt=dt)
# simulate without for-loop
y_meas, x_state = pf.simulate(bm_model, key, n_obs, x_init, theta)

# particle filter specification
n_particles = 7
key, subkey = random.split(key)
# # pf with for-loop
pf_out1 = pf.particle_filter_for(
    bm_model, subkey, y_meas, theta, n_particles)
# pf without for-loop
pf_out2 = pf.particle_filter(
    bm_model, subkey, y_meas, theta, n_particles)

max_diff = {
    k: jnp.max(jnp.abs(pf_out1[k] - pf_out2[k]))
    for k in pf_out1.keys()
}
