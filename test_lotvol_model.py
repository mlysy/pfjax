import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random

exec(open("lotvol_model.py").read())

# parameter values
alpha = 1.1
beta = 1.2
gamma = 4.
delta = 1.3
sigma_H = .1
sigma_L = .2
tau_H = .25
tau_L = .35
theta = jnp.array([alpha, beta, gamma, delta, sigma_H, sigma_L, tau_H, tau_L])

dt = .44
n_res = 3
n_state = (n_res, 2)

key = random.PRNGKey(0)

key, subkey = random.split(key)
x_prev = random.normal(subkey, n_state)

key, subkey = random.split(key)
x_curr = state_sample(x_prev, theta, key)
x_curr_for = state_sample_for(x_prev, theta, key)

print("x_curr - x_curr_for = \n", x_curr - x_curr_for)

breakpoint()


state_lpdf(x_curr, x_prev, theta)
