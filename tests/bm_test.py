""" not updated for new file structure yet. Imports may not work """

## test file for brownian motion model:
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import pandas as pd
import seaborn as sns

import particle_filter as pf
import particle_filter_mvn as pfmvn
import bm_model as bm
from proj_data import proj_data

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
y_meas, x_state = pf.meas_sim(bm_model, n_obs, x_init, theta, subkey)

# plot data
plot_df = (pd.DataFrame({"time": jnp.arange(n_obs) * dt,
                         "x_state": jnp.squeeze(x_state),
                         "y_meas": jnp.squeeze(y_meas)})
           .melt(id_vars="time", var_name="type"))
sns.relplot(
    data=plot_df, kind="line",
    x="time", y="value", hue="type"
)


# particle filter specification
n_particles = 10

# # timing without jit
key, subkey = random.split(key)

# test pf_MVN:
""" particle_filter_for and particle_filter should return the same values """

pffor = pfmvn.particle_filter_for(
    bm_model, y_meas, theta, n_particles, subkey)
print("Pf for: \n")
print(pffor["logw_particles"][-1])
print(pffor["X_particles"][85:])

mvn = pfmvn.particle_filter(
    bm_model, y_meas, theta, n_particles, subkey)
print("Pf mvn: \n")
print(mvn["logw_particles"][-1])
print(mvn["X_particles_mu"][85:])
