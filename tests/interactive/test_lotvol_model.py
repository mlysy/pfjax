import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random

exec(open("lotvol_model.py").read())
exec(open("particle_filter.py").read())

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

dt = .1
n_res = 10
n_state = (n_res, 2)

key = random.PRNGKey(0)

key, subkey = random.split(key)
x_prev = random.normal(subkey, n_state)

key, subkey = random.split(key)
x_curr = state_sample(x_prev, theta, key)
x_curr_for = state_sample_for(x_prev, theta, key)

print("x_curr - x_curr_for = \n", x_curr - x_curr_for)

state_lp = state_lpdf(x_curr, x_prev, theta)
print("state_lp = \n", state_lp)


# --- particle filter ----------------------------------------------------------

key, subkey = random.split(key)
x_init = init_sample(y_init=jnp.log(jnp.array([5., 3.])),
                     theta=jnp.append(theta[0:6], jnp.array([0., 0.])),
                     key=subkey)

n_obs = 100
key, subkey = random.split(key)
y_meas, x_state = meas_sim(n_obs, x_init, theta, subkey)
t_seq = jnp.arange(n_obs) * dt

plt.plot(t_seq, y_meas[:, 0])
# plt.show()

n_particles = 100
pf_out = particle_filter(y_meas, theta, n_particles, key)
