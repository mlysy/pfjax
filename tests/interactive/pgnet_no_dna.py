import numpy as np
import pandas as pd
import seaborn as sns

import jax
import jax.numpy as jnp
import jax.random as random

import pfjax as pf
from pfjax import models

key = random.PRNGKey(0)

theta = np.array([0.1, 0.7, 0.35, 0.2, 0.1, 0.9, 0.3, 0.1])
tau = np.array([1,1,1])*1

DNA_init = np.array([5])

theta = np.concatenate([theta, tau, DNA_init])
dt = .1
n_res = 1
n_obs = 50

pgnet_dna = models.PGNETModelNoDNA(dt, n_res)
pgnet_dna_bridge = models.PGNETModelNoDNA(dt, n_res, bootstrap = False)

key, subkey = random.split(key)

x_init = jnp.block([[jnp.zeros((n_res-1, 4))],
                   [jnp.log(jnp.array([8., 8., 8., 5.]))]])

y_meas, x_state = pf.simulate(pgnet_dna, subkey, n_obs, x_init, theta)
print("Running bootstrap pf")

# tmp = pf.particle_filter(
#         theta=theta, 
#         model=pgnet_dna, 
#         y_meas=y_meas, 
#         n_particles=10, 
#         key=key,
#         particle_sampler = pf.particle_resample_mvn)

print("Ran bootstrap pf")

print(pgnet_dna.meas_lpdf(y_curr = y_meas[0], x_curr = x_state[0], theta = theta))


print("Bridge pf")

pgnet_dna_bridge = models.PGNETModelNoDNA(dt, n_res, bootstrap = False)

tmp = pf.particle_filter(
        theta=theta, 
        model=pgnet_dna_bridge, 
        y_meas=y_meas, 
        n_particles=100, 
        key=key,
        particle_sampler = pf.particle_resample_mvn)

print(tmp)