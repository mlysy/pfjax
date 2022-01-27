import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import pfjax as pf

breakpoint()


def particle_resample_mvn(key, x_particles_prev, logw):
    wgt = jnp.exp(logw - jnp.max(logw))
    prob = wgt / jnp.sum(wgt)
    p_shape = x_particles_prev.shape  # should be (n_particles, ...)
    n_particles = p_shape[0]
    # calculate weighted mean and variance
    x_particles = jnp.transpose(x_particles_prev.reshape((n_particles, -1)))
    mvn_mean = jnp.average(x_particles, axis=1, weights=prob)
    mvn_cov = jnp.atleast_2d(jnp.cov(x_particles, aweights=prob))
    # sample from the mvn
    x_particles = random.multivariate_normal(key,
                                             mean=mvn_mean,
                                             cov=mvn_cov,
                                             shape=(n_particles,))
    return {
        "x_particles": jnp.reshape(x_particles, newshape=p_shape),
        "mean": mvn_mean,
        "cov": mvn_cov
    }


key = random.PRNGKey(0)
# parameter values
mu = 5
sigma = 1
tau = .1
theta = jnp.array([mu, sigma, tau])
# data specification
dt = .1
n_obs = 5
x_init = jnp.array(0.)
bm_model = pf.BMModel(dt=dt)
# simulate without for-loop
y_meas, x_state = pf.simulate(bm_model, key, n_obs, x_init, theta)

# # check jit
# y_meas, x_state = jax.jit(pf.simulate, static_argnums=(0, 2))(
#     bm_model, key, n_obs, x_init, theta)

# particle filter specification
n_particles = 7
key, subkey = random.split(key)
# # pf with for-loop
pf_out1 = pf.particle_filter_for(
    bm_model, subkey, y_meas, theta, n_particles)
# particle_resample_mvn(key,
#                       x_particles_prev=pf_out1["x_particles"][1],
#                       logw=pf_out1["logw"][1])
# pf without for-loop
pf_out2 = pf.particle_filter(
    bm_model, subkey, y_meas, theta, n_particles)

breakpoint()

max_diff = {
    k: jnp.max(jnp.abs(pf_out1[k] - pf_out2[k]))
    for k in pf_out1.keys()
}

max_diff
