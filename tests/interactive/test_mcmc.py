import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import pfjax as pf
import pfjax.mcmc as mcmc
from pfjax.models import BMModel
# import particle_filter as pf
# import bm_model as bm


key = random.PRNGKey(0)
# parameter values
mu, sigma, tau = 5., 1., .1
theta = jnp.array([mu, sigma, tau])
# data specification
dt = .1
n_obs = 10
x_init = jnp.array(0.)
bm_model = BMModel(dt=dt)
# simulate
key, subkey = random.split(key)
y_meas, x_state = pf.simulate(bm_model, subkey, n_obs, x_init, theta)

# # loglikelihood check
# loglik1 = mcmc.full_loglik_for(bm_model, y_meas, x_state, theta)
# loglik2 = mcmc.full_loglik(bm_model, y_meas, x_state, theta)

# print("loglik1 = {}".format(loglik1))
# print("loglik2 = {}".format(loglik2))


# mwg update
prior = NormalDiagPrior(loc=theta, scale=jnp.abs(theta))
rw_sd = jnp.array([.1] * theta.size)

# # with default order
# key = random.PRNGKey(0)
# theta_order = jnp.arange(theta.size)
# key, subkey = random.split(key)
# theta1 = mcmc.param_mwg_update_for(bm_model, prior, subkey, theta,
#                                    x_state, y_meas, rw_sd, theta_order)
# theta2 = mcmc.param_mwg_update(bm_model, prior, subkey, theta,
#                                x_state, y_meas, rw_sd, theta_order)
# print("theta1 = {}".format(theta1))
# print("theta2 = {}".format(theta2))

# with non-default order
key = random.PRNGKey(0)
rw_sd = jnp.array([.1] * theta.size)
key, subkey = random.split(key)
n_updates = 10
theta_order = random.choice(subkey, jnp.arange(theta.size), shape=(n_updates,))
key, subkey = random.split(key)
theta1 = mcmc.param_mwg_update_for(bm_model, prior, subkey, theta,
                                   x_state, y_meas, rw_sd, theta_order)
theta2 = mcmc.param_mwg_update(bm_model, prior, subkey, theta,
                               x_state, y_meas, rw_sd, theta_order)
print("theta_order = {}".format(theta_order))
print("theta1 = {}".format(theta1))
print("theta2 = {}".format(theta2))

breakpoint()


# -------------------------------------------------------------------------------
# ok let's see if it's possible to jit-compile functions through **kwargs.

def bar(x, y):
    return x + y


def mcmc_update(theta, logpost, logpost_args={}):
    return logpost(theta, **logpost_args)


@jax.jit
def baz(x, y):
    return mcmc_update(theta=x, logpost=bar, logpost_args={"y": y})


baz(x=3., y=5.)

# ok can we jit foo directly?

mcmc_update_jit = jax.jit(mcmc_update, static_argnums=1)

mcmc_update_jit(x=3., bar=bar, bar_args={"y": 5.})
