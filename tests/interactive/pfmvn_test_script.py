""" Unit test for pfmvn """

## test file for brownian motion model:
from pfjax import stoch_opt, get_sum_lweights, get_sum_lweights_mvn
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

import pfjax as pf
import pfjax.particle_filter as particle_filter
from pfjax.particle_filter import particle_resample_mvn
from pfjax import LotVolModel
import pfjax.particle_filter_mvn as pfmvn


# initial key for random numbers
key = random.PRNGKey(0)

# # parameter values
# mu = 5.
# sigma = .2
# tau = 1.
# theta = np.array([mu, sigma, tau])

# # data specification
# dt = .2
# n_obs = 100
# x_init = jnp.array([0.])

# # simulate data
# bm_model = bm.BMModel(dt=dt)
# key, subkey = random.split(key)
# y_meas, x_state = simulate(bm_model, n_obs, x_init, theta, subkey)

# # particle filter specification
# n_particles = 10

# # # timing without jit
# key, subkey = random.split(key)

# # test pf_MVN:
# """ particle_filter_for and particle_filter should return the same values """

# start = time.perf_counter()
# pffor = pfmvn.particle_filter_for(bm_model,
#                                    y_meas,
#                                    theta, 
#                                    n_particles, 
#                                    subkey)
# print("bm model pf-for time: {0}".format(time.perf_counter() - start))
# del(start) # just to be safe

# start = time.perf_counter()
# mvn = pfmvn.particle_filter(bm_model,
#                              y_meas,
#                              theta, 
#                              n_particles, 
#                              subkey)
# print("bm model pf time: {0}".format(time.perf_counter() - start))
# del(start)

# assert(mvn["logw_particles"].all() == pffor["logw_particles"].all())
# # assert(mvn["X_particles_mu"].all() == pffor["X_particles_mu"].all())


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
n_res = 2
n_obs = 50
lotvol_model = LotVolModel(dt, n_res)
key, subkey = random.split(key)


x_init = jnp.block([[jnp.zeros((n_res-1, 2))],
                    [jnp.log(jnp.array([5., 3.]))]])

y_meas, x_state = pf.simulate(model=lotvol_model,
                              n_obs=n_obs,
                              x_init=x_init,
                              theta=theta,
                              key=subkey)


n_particles = 3
# lv_pffor = pfmvn.particle_filter_for(lotvol_model,
#                                   y_meas,
#                                   theta,
#                                   n_particles,
#                                   subkey)

# lv_mvn = particle_filter(theta=theta,
#                                model=lotvol_model,
#                                y_meas=y_meas,
#                                n_particles=n_particles, particle_sampler = particle_resample_mvn,
#                                key=subkey)

# print(lv_mvn["logw"][:5])
# print(lv_mvn["x_particles"][:3])

# assert(lv_mvn["logw_particles"].all() == lv_pffor["logw_particles"].all())
# assert(lv_mvn["X_particles_mu"].all() == lv_pffor["X_particles_mu"].all())

params = stoch_opt(model = lotvol_model, 
                   params = jnp.array([1.,1., 4., 1., 0.1, 0.1, 0.25, 0.25]), 
                   grad_fun = get_sum_lweights_mvn, 
                   y_meas = y_meas, key=key, 
                   n_particles = 10,
                   learning_rate=1e-5, 
                   iterations=10,
                   mask=np.array([1,1,1,1,1,1,0, 0]))
print(params)