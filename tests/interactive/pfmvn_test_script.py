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
 
import pfjax as pf
from pfjax.particle_filter import particle_filter, particle_loglik, particle_resample_mvn
import pfjax.sde
# from pfjax import stoch_opt, get_sum_lweights, get_sum_lweights_mvn


# FIXME: temp stoch opt code: 
import optax
from functools import partial


def get_sum_lweights(theta, key, n_particles, y_meas, model, **kwargs):
    ret = particle_filter(model, key, y_meas, theta, n_particles)
    sum_particle_lweights = particle_loglik(ret['logw'])
    return sum_particle_lweights

def get_sum_lweights_mvn(theta, key, n_particles, y_meas, model):
    ret = particle_filter(model=model,
                          y_meas=y_meas, 
                          theta=theta, 
                          n_particles=n_particles,
                          key=key, 
                          particle_sampler=particle_resample_mvn)
    sum_particle_lweights = particle_loglik(ret['x_particles'], ret['logw'])
    # breakpoint()
    return sum_particle_lweights


def update_params(params, subkey, opt_state, grad_fun=None, n_particles=100, y_meas=None, model=None, learning_rate=0.01, mask=None,
                  optimizer=None, **kwargs):
    params_update = jax.grad(grad_fun, argnums=0)(
        params, subkey, n_particles, y_meas, model)
    params_update = jnp.where(mask, params_update, 0)
    updates, opt_state = optimizer.update(params_update, opt_state)
    return optax.apply_updates(params, updates)


def stoch_opt(model, params, grad_fun, y_meas, n_particles=100, iterations=10,
              learning_rate=0.01, key=1, mask=None):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    partial_update_params = partial(update_params, n_particles=n_particles, y_meas=y_meas,
                                    model=model, learning_rate=learning_rate, mask=mask, grad_fun=grad_fun, optimizer=optimizer)
    update_fn = jax.jit(partial_update_params, donate_argnums=(0,))
    keys = random.split(key, iterations)
    for i, subkey in enumerate(keys):
        _loglik = grad_fun(params, subkey, n_particles, y_meas, model)
        print(
            "Stoch_opt iteration: {0}. Log-likelihood: {1}".format(i, _loglik))
        # breakpoint()
        params = update_fn(params, subkey, opt_state)
        # print(params)
    return params


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
                   grad_fun=get_sum_lweights_mvn,
                   y_meas = y_meas, key=key, 
                   n_particles=n_particles,
                   learning_rate=1e-5, 
                   iterations=10,
mask=np.array([1,1,1,1,1,1,1,1]))
print(params)


## 
# def _lweight_to_prob(logw):
#     wgt = jnp.exp(logw - jnp.max(logw))
#     prob = wgt / jnp.sum(wgt)
#     return prob


# def particle_resample_mvn_for(key, x_particles_prev, logw):
#     n_particles, n_res, n_states = x_particles_prev.shape
#     n_dim = n_res * n_states
#     prob = _lweight_to_prob(logw)
#     flat = x_particles_prev.reshape((n_particles, n_dim))
#     mu = jnp.average(flat, axis=0, weights=prob)
#     cov_mat = jnp.zeros((n_dim, n_dim))
#     for i in range(n_dim):
#         for j in range(i, n_dim):
#             c = jnp.cov(flat[:, i], flat[:, j], aweights=prob)
#             cov_mat = cov_mat.at[i, j].set(c[0][1])
#             cov_mat = cov_mat.at[j, i].set(cov_mat[i, j])
#     cov_mat += jnp.diag(jnp.ones(n_dim) * 1e-10)  # for numeric stability
#     print("Is positive definite?: ", np.all(np.linalg.eigvals(cov_mat) >= 0))
#     samples = random.multivariate_normal(key,
#                                          mean=mu,
#                                          cov=cov_mat,
#                                          shape=(n_particles,))
#     ret_val = {"x_particles": samples.reshape(x_particles_prev.shape),
#                "x_particles_mu": mu,
#                "cov_mat": cov_mat}
#     return ret_val


# def particle_resample_mvn_martin(key, x_particles_prev, logw):
#     cont_key = random.PRNGKey(0)
#     wgt = jnp.exp(logw - jnp.max(logw))
#     prob = wgt / jnp.sum(wgt)
#     p_shape = x_particles_prev.shape  # should be (n_particles, ...)
#     n_particles = p_shape[0]
#     # calculate weighted mean and variance
#     x_particles = jnp.transpose(x_particles_prev.reshape((n_particles, -1)))
#     mvn_mean = jnp.average(x_particles, axis=1, weights=prob)
#     mvn_cov = jnp.atleast_2d(jnp.cov(x_particles, aweights=prob))
#     mvn_cov += jnp.diag(jnp.ones(p_shape[1]*p_shape[2]) * 1e-10)  # for numeric stability
#     # sample from the mvn
#     x_particles = random.multivariate_normal(cont_key,
#                                             mean=mvn_mean,
#                                             cov=mvn_cov,
#                                             shape=(n_particles,))
#     return {
#         "x_particles": jnp.reshape(x_particles, newshape=p_shape),
#         "mean": mvn_mean,
#         "cov": mvn_cov
#     }

# key, *subkeys = random.split(key, num=n_particles+1)
# x_particles, logw = jax.vmap(
#     lambda k: lotvol_model.pf_init(k, y_meas[0], 
#                                    jnp.array([1., 1., 4., 1., 0.1, 0.1, 0.25, 0.25])))(jnp.array(subkeys))

# f1 = particle_resample_mvn_for(random.PRNGKey(0), x_particles, logw)
# f4 = particle_resample_mvn_martin(key, x_particles, logw)
# # p1, logw1 = jax.vmap(lambda xs, k: lotvol_model.pf_step(k, xs, y_meas[1], theta))(f1["x_particles"], jnp.array(subkeys))
# breakpoint()
