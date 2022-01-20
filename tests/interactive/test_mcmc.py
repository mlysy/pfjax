import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import pfjax as pf
import mcmc
# import particle_filter as pf
# import bm_model as bm

key = random.PRNGKey(0)
# parameter values
mu = 5
sigma = 1
tau = .1
theta = jnp.array([mu, sigma, tau])
# data specification
dt = .1
n_obs = 10
x_init = jnp.array([0.])
bm_model = pf.BMModel(dt=dt)
# simulate
key, subkey = random.split(key)
y_meas, x_state = pf.simulate(bm_model, n_obs, x_init, theta, subkey)

# loglikelihood check
loglik1 = mcmc.full_loglik_for(bm_model, y_meas, x_state, theta)
loglik2 = mcmc.full_loglik(bm_model, y_meas, x_state, theta)

print("loglik1 = {}".format(loglik1))
print("loglik2 = {}".format(loglik2))


# mwg update
prior = mcmc.FlatPrior()
rw_sd = jnp.array([.01] * theta.size)
key, subkey = random.split(key)
theta1 = mcmc.param_mwg_update_for(bm_model, prior, theta,
                                   x_state, y_meas, rw_sd, subkey)
theta2 = mcmc.param_mwg_update(bm_model, prior, theta,
                               x_state, y_meas, rw_sd, subkey)

print("theta1 = {}".format(theta1))
print("theta2 = {}".format(theta2))

breakpoint()
