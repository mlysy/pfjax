""" Unit test for pfmvn """

## test file for brownian motion model:
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

from pfjax.particle_filter import particle_filter, simulate
import pfjax.bm_model as bm
from pfjax.lotvol_model import LotVolModel
import pfjax.particle_filter_mvn as pfmvn

import time

# initial key for random numbers
key = random.PRNGKey(0)

# parameter values
mu = 5.
sigma = .2
tau = 1.
theta = np.array([mu, sigma, tau])

# data specification
dt = .2
n_obs = 100
x_init = jnp.array([0.])

# simulate data
bm_model = bm.BMModel(dt=dt)
key, subkey = random.split(key)
y_meas, x_state = simulate(bm_model, n_obs, x_init, theta, subkey)

# particle filter specification
n_particles = 10

# # timing without jit
key, subkey = random.split(key)

# test pf_MVN:
""" particle_filter_for and particle_filter should return the same values """

start = time.perf_counter()
pffor = pfmvn.particle_filter_for(bm_model,
                                   y_meas,
                                   theta, 
                                   n_particles, 
                                   subkey)
print("bm model pf-for time: {0}".format(time.perf_counter() - start))
del(start) # just to be safe

start = time.perf_counter()
mvn = pfmvn.particle_filter(bm_model,
                             y_meas,
                             theta, 
                             n_particles, 
                             subkey)
print("bm model pf time: {0}".format(time.perf_counter() - start))
del(start)

assert(mvn["logw_particles"].all() == pffor["logw_particles"].all())
# assert(mvn["X_particles_mu"].all() == pffor["X_particles_mu"].all())


#### lotvol tests
key = random.PRNGKey(0)

# parameter values

alpha = 1.0
beta = 1.0
gamma = 4.0
delta = 1.0
sigma_h = 0.1
sigma_l = 0.1
tau_h = 0.25  # 0.1 for low noise
tau_l = 0.25 # 0.1 for low noise

theta = np.array([alpha, beta, gamma, delta, sigma_h, sigma_l, tau_h, tau_l])

dt = 0.1
n_res = 1
n_obs = 50
lotvol_model = LotVolModel(dt, n_res)
key, subkey = random.split(key)


x_init = lotvol_model.init_sample(y_init=jnp.log(jnp.array([5., 3.])),
                                  theta=jnp.append(
                                      theta[0:6], jnp.array([0., 0.])),
                                  key=subkey)
y_meas, x_state = simulate(lotvol_model, n_obs, x_init, theta, subkey)


n_particles = 10
# lv_pffor = pfmvn.particle_filter_for(lotvol_model,
#                                   y_meas,
#                                   theta,
#                                   n_particles,
#                                   subkey)

start = time.perf_counter()
lv_mvn = pfmvn.particle_filter(lotvol_model,
                            y_meas,
                            theta,
                            n_particles,
                            subkey)
print("lotvol mvn time: {0}".format(time.perf_counter() - start))
print(lv_mvn["logw_particles"][:5])

# assert(lv_mvn["logw_particles"].all() == lv_pffor["logw_particles"].all())
# assert(lv_mvn["X_particles_mu"].all() == lv_pffor["X_particles_mu"].all())
