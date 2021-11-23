import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random

# hack to copy-paste in contents without import
exec(open("bm_model.py").read())
exec(open("particle_filter.py").read())


key = random.PRNGKey(0)


# parameter values
mu = 5
sigma = 1
tau = .1
theta = jnp.array([mu, sigma, tau])

print(theta)

# data specification
dt = .1
n_obs = 5
x_init = jnp.array([0.])

# simulate regular data
y_meas_for, x_state_for = meas_sim_for(n_obs, x_init, theta, key)

# simulate lax data
y_meas, x_state = meas_sim(n_obs, x_init, theta, key)

print("max_diff between sim_for and sim_scan:\n")
print("y_meas = \n",
      jnp.max(jnp.abs(y_meas_for - y_meas)))
print("x_state = \n",
      jnp.max(jnp.abs(x_state_for - x_state)))

# run particle filter
n_particles = 7
key, subkey = random.split(key)
pf_for = particle_filter_for(y_meas, theta, n_particles, subkey)
pf_scan = particle_filter(y_meas, theta, n_particles, subkey)

print("max_diff between pf_for and pf_scan:\n")
for k in pf_for.keys():
    print(k, " = \n",
          jnp.max(jnp.abs(pf_for[k] - pf_scan[k])))
