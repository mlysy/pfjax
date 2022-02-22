""" Unit test for pfmvn """

## test file for brownian motion model:
from pfjax import stoch_opt
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

import pfjax as pf
from pfjax import LotVolModel 
from pfjax.particle_filter import particle_filter, particle_loglik, particle_resample_mvn, particle_neg_loglik


# FIXME: temp stoch opt code: 
import optax
from functools import partial


# #### lotvol tests
key = random.PRNGKey(0)

# parameter values
alpha = 1.0
beta = 1.0
gamma = 4.0
delta = 1.0
sigma_h = 0.1
sigma_l = 0.1
tau_h = 0.25  # low noise = 0.1
tau_l = 0.25  # low noise = 0.1

theta = np.array([alpha, beta, gamma, delta, sigma_h, sigma_l, tau_h, tau_l])

dt = 0.1
n_res = 5
n_obs = 50
lotvol_model = LotVolModel(dt, n_res)

key = random.PRNGKey(0)
key, subkey = random.split(key)

x_init = jnp.block([[jnp.zeros((n_res-1, 2))],
                    [jnp.log(jnp.array([5., 3.]))]])

y_meas, x_state = pf.simulate(model=lotvol_model,
                              n_obs=n_obs,
                              x_init=x_init,
                              theta=theta,
                              key=subkey)

n_particles = 50

# breakpoint()

params = stoch_opt(model = lotvol_model, 
                #    params = jnp.array([1.,1., 4., 1., 0.1, 0.1, 0.25, 0.25]), 
                   params = jnp.array([1.00001, 1.00001, 3.99999, 1.00001, 0.09999, 0.09999, 0.24999, 0.25001]),
                   grad_fun=particle_neg_loglik,
                   y_meas = y_meas, key=key, 
                   n_particles=n_particles,
                   learning_rate=1e-5, 
                   iterations=10,
mask=np.array([1,1,1,1,1,1,1,1]))
print(params)

