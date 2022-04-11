import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
from functools import partial
import pfjax as pf
from pfjax.models import BMModel
from pfjax.particle_filter import _lweight_to_prob


def abs_err(x1, x2):
    return jnp.max(jnp.abs(x1-x2))


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
bm_model = BMModel(dt=dt)
# simulate without for-loop
y_meas, x_state = pf.simulate(bm_model, key, n_obs, x_init, theta)

# particle filter specification
n_particles = 7
key, subkey = random.split(key)
# # pf with for-loop
pf_out1 = pf.particle_filter_for(
    bm_model, subkey, y_meas, theta, n_particles)
# pf without for-loop
pf_out2 = pf.particle_filter(
    bm_model, subkey, y_meas, theta, n_particles)

max_diff = {
    k: jnp.max(jnp.abs(pf_out1[k] - pf_out2[k]))
    for k in pf_out1.keys()
}
print(max_diff)


acc_dict = True


def accumulate_score(x_prev, x_curr, y_curr, theta):
    r"""
    Accumulator for score function.
    """
    measgrad_lpdf = jax.grad(bm_model.meas_lpdf, argnums=2)
    stategrad_lpdf = jax.grad(bm_model.state_lpdf, argnums=2)
    if not acc_dict:
        return measgrad_lpdf(y_curr, x_curr, theta) + \
            stategrad_lpdf(x_curr, x_prev, theta)
    else:
        return {"meas": measgrad_lpdf(y_curr, x_curr, theta),
                "state": stategrad_lpdf(x_curr, x_prev, theta)}


# new pf
pf_out3 = pf.particle_filter2(
    bm_model, subkey, y_meas, theta, n_particles,
    history=True, accumulator=accumulate_score)

# check x_particles and logw
max_diff = {k: abs_err(pf_out2[k], pf_out3[k])
            for k in ["x_particles", "logw"]}
print(max_diff)

# check ancestors
max_diff = {k: abs_err(pf_out2[k], pf_out3["resample_out"][k])
            for k in ["ancestors"]}
print(max_diff)

# check loglik
max_diff = abs_err(pf.particle_loglik(pf_out2["logw"]), pf_out3["loglik"])
print(max_diff)

# new pf without history
pf_out4 = pf.particle_filter2(
    bm_model, subkey, y_meas, theta, n_particles,
    history=False, accumulator=accumulate_score)

# check x_particles and logw
max_diff = {k: abs_err(pf_out2[k][n_obs-1],  pf_out4[k])
            for k in ["x_particles", "logw"]}
print(max_diff)

# check ancestors
max_diff = {k: abs_err(pf_out2[k][n_obs-1], pf_out4["resample_out"][k])
            for k in ["ancestors"]}
print(max_diff)

# check loglik
max_diff = abs_err(pf.particle_loglik(pf_out2["logw"]), pf_out4["loglik"])
print(max_diff)

# check accumulator
if acc_dict:
    max_diff = {k: abs_err(jnp.sum(pf_out3["accumulate_out"][k][n_obs-2] *
                                   jnp.atleast_2d(_lweight_to_prob(
                                       pf_out3["logw"][n_obs-1])).T,
                                   axis=0),
                           pf_out4["accumulate_out"][k])
                for k in ["state", "meas"]}
else:
    max_diff = abs_err(jnp.sum(pf_out3["accumulate_out"][n_obs-2] *
                               jnp.atleast_2d(_lweight_to_prob(
                                   pf_out3["logw"][n_obs-1])).T,
                               axis=0),
                       pf_out4["accumulate_out"])
print(max_diff)

# --- test accumulator ---------------------------------------------------------
