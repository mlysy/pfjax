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

# --- check for-loop -----------------------------------------------------------

if False:
    # pf with for-loop
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

# --- check pf v2 with history -------------------------------------------------

if False:
    # old pf without for-loop
    pf_out1 = pf.particle_filter(
        bm_model, subkey, y_meas, theta, n_particles)
    # new pf with history
    pf_out2 = pf.particle_filter2(
        bm_model, subkey, y_meas, theta, n_particles,
        history=True)

    # check x_particles and logw
    max_diff = {k: abs_err(pf_out1[k], pf_out2[k])
                for k in ["x_particles", "logw"]}
    print(max_diff)

    # check ancestors
    max_diff = {k: abs_err(pf_out1[k], pf_out2["resample_out"][k])
                for k in ["ancestors"]}
    print(max_diff)

    # check loglik
    max_diff = abs_err(pf.particle_loglik(pf_out1["logw"]), pf_out2["loglik"])
    print(max_diff)

# --- check pf v2 without history ----------------------------------------------

if False:
    # old pf without for-loop
    pf_out1 = pf.particle_filter(
        bm_model, subkey, y_meas, theta, n_particles)
    # new pf without history
    pf_out2 = pf.particle_filter2(
        bm_model, subkey, y_meas, theta, n_particles,
        history=False)

    # check x_particles and logw
    max_diff = {k: abs_err(pf_out1[k][n_obs-1],  pf_out2[k])
                for k in ["x_particles", "logw"]}
    print(max_diff)

    # check ancestors
    max_diff = {k: abs_err(pf_out1[k][n_obs-1], pf_out2["resample_out"][k])
                for k in ["ancestors"]}
    print(max_diff)

    # check loglik
    max_diff = abs_err(pf.particle_loglik(pf_out1["logw"]), pf_out2["loglik"])
    print(max_diff)


# --- test accumulator ---------------------------------------------------------

if False:
    def accumulate_ancestors(x_prev, x_curr, y_curr, theta):
        r"""
        Returns just x_prev and x_curr to check that ancestors are being computed as expected.
        """
        return x_prev, x_curr

    # new pf with history
    pf_out1 = pf.particle_filter2(
        bm_model, subkey, y_meas, theta, n_particles,
        history=True, accumulator=accumulate_ancestors)

    # check ancestors
    max_diff = []
    for i in range(n_obs-1):
        max_diff += [abs_err(
            pf_out1["x_particles"][i, pf_out1["resample_out"]["ancestors"][i]],
            pf_out1["accumulate_out"][0][i])]
    max_diff = jnp.array(max_diff)
    print(max_diff)


if True:

    def accumulate_ancestors(x_prev, x_curr, y_curr, theta):
        r"""
        Returns just x_prev and x_curr to check that ancestors are being computed as expected.
        """
        return x_prev, x_curr

    def accumulate_score(x_prev, x_curr, y_curr, theta):
        r"""
        Accumulator for score function.
        """
        measgrad_lpdf = jax.grad(bm_model.meas_lpdf, argnums=2)
        stategrad_lpdf = jax.grad(bm_model.state_lpdf, argnums=2)
        return measgrad_lpdf(y_curr, x_curr, theta) + stategrad_lpdf(x_curr, x_prev, theta)
            
        # return {"meas": measgrad_lpdf(y_curr, x_curr, theta),
        #         "state": stategrad_lpdf(x_curr, x_prev, theta)}

    # def brute_force_acc(x_prev, x_curr, y_curr, theta):
    #     measgrad_lpdf = jax.grad(bm_model.meas_lpdf, argnums=2)
    #     stategrad_lpdf = jax.grad(bm_model.state_lpdf, argnums=2)
    #     t1 = measgrad_lpdf(y_curr, x_curr, theta)
    #     t2 = stategrad_lpdf(x_curr, x_prev, theta)
    #     return t1 + t2

    # new pf with history
    pf_out1 = pf.particle_filter2(
        bm_model, subkey, y_meas, theta, n_particles,
        history=True, accumulator=accumulate_score)

    # check brute force calculation

    Xt_1n = pf_out1["x_particles"]
    # Now, we compute G_n = G(Xt_1n)
    tot = 0
    for i in range(1, Xt_1n.shape[0] - 1):
        tot += brute_force_acc(Xt_1n[i - 1,:], Xt_1n[i,:], y_meas[i], theta)
        breakpoint()


