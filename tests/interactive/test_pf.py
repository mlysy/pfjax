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


def get_particles(i_part, x_particles, ancestors):
    """
    Return a full particle by backtracking through ancestors of particle `i_part` at last time point.
    """
    n_obs = x_particles.shape[0]

    # scan function
    def get_ancestor(i_part_next, t):
        # ancestor particle index
        i_part_curr = ancestors[t, i_part_next]
        res = i_part_curr
        return res, res

    # scan initial value
    i_part_init = i_part
    # lax.scan itself
    last, full = jax.lax.scan(get_ancestor, i_part_init,
                              jnp.arange(n_obs-1), reverse=True)
    i_part_full = jnp.concatenate([full, jnp.array(i_part_init)[None]])
    return x_particles[jnp.arange(n_obs), i_part_full, ...]  # , i_part


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
    max_diff = {
        "loglik": abs_err(pf.particle_loglik(pf_out1["logw"]),
                          pf_out2["loglik"])
    }
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
    max_diff = {
        "loglik": abs_err(pf.particle_loglik(pf_out1["logw"]),
                          pf_out2["loglik"])
    }
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
    ancestors1 = []
    ancestors2 = []
    for i in range(n_obs-1):
        ancestors1 += [pf_out1["x_particles"]
                       [i, pf_out1["resample_out"]["ancestors"][i]]]
        ancestors2 += [pf_out1["accumulate_out"][0][i]]
    ancestors1 = jnp.array(ancestors1)
    ancestors2 = jnp.array(ancestors2)
    max_diff = {
        "ancestors_acc": abs_err(ancestors1, ancestors2)
    }
    print(max_diff)


if False:
    def accumulate_score(x_prev, x_curr, y_curr, theta):
        r"""
        Accumulator for score function.
        """
        grad_meas = jax.grad(bm_model.meas_lpdf, argnums=2)
        grad_state = jax.grad(bm_model.state_lpdf, argnums=2)
        return grad_meas(y_curr, x_curr, theta) + \
            grad_state(x_curr, x_prev, theta)

    # new pf with history
    pf_out1 = pf.particle_filter2(
        bm_model, subkey, y_meas, theta, n_particles,
        history=True, accumulator=accumulate_score)

    # new pf without history
    pf_out2 = pf.particle_filter2(
        bm_model, subkey, y_meas, theta, n_particles,
        history=False, accumulator=accumulate_score)

    # brute force calculation
    x_particles = pf_out1["x_particles"]
    ancestors = pf_out1["resample_out"]["ancestors"]
    prob = _lweight_to_prob(pf_out1["logw"][n_obs-1])
    x_particles_full = jax.vmap(
        lambda i: get_particles(i, x_particles, ancestors)
    )(jnp.arange(n_particles))
    x_particles_prev = x_particles_full[:, :-1]
    x_particles_curr = x_particles_full[:, 1:]
    y_curr = y_meas[1:]
    acc_out = jax.vmap(
        jax.vmap(
            accumulate_score,
            in_axes=(0, 0, 0, None)
        ),
        in_axes=(0, 0, None, None)
    )(x_particles_prev, x_particles_curr, y_curr, theta)
    acc_out = acc_out.transpose((1, 0, 2))
    acc_out = jnp.sum(
        jax.vmap(jnp.multiply)(jnp.sum(acc_out, axis=0), prob),
        axis=0
    )
    max_diff = {
        "score_acc": abs_err(acc_out, pf_out2["accumulate_out"])
    }
    print(max_diff)

if True:
    def accumulate_hessian(x_prev, x_curr, y_curr, theta):
        r"""
        Accumulator for hessian function.
        """
        hess_meas = jax.jacfwd(jax.jacrev(bm_model.meas_lpdf, argnums=2),
                               argnums=2)
        hess_state = jax.jacfwd(jax.jacrev(bm_model.state_lpdf, argnums=2),
                                argnums=2)
        return hess_meas(y_curr, x_curr, theta) + \
            hess_state(x_curr, x_prev, theta)

    # new pf with history
    pf_out1 = pf.particle_filter2(
        bm_model, subkey, y_meas, theta, n_particles,
        history=True, accumulator=accumulate_hessian)

    # new pf without history
    pf_out2 = pf.particle_filter2(
        bm_model, subkey, y_meas, theta, n_particles,
        history=False, accumulator=accumulate_hessian)

    # brute force calculation
    x_particles = pf_out1["x_particles"]
    ancestors = pf_out1["resample_out"]["ancestors"]
    prob = _lweight_to_prob(pf_out1["logw"][n_obs-1])
    x_particles_full = jax.vmap(
        lambda i: get_particles(i, x_particles, ancestors)
    )(jnp.arange(n_particles))
    x_particles_prev = x_particles_full[:, :-1]
    x_particles_curr = x_particles_full[:, 1:]
    y_curr = y_meas[1:]
    acc_out = jax.vmap(
        jax.vmap(
            accumulate_hessian,
            in_axes=(0, 0, 0, None)
        ),
        in_axes=(0, 0, None, None)
    )(x_particles_prev, x_particles_curr, y_curr, theta)
    acc_out = acc_out.transpose((1, 0, 2, 3))
    acc_out = jnp.sum(
        jax.vmap(jnp.multiply)(jnp.sum(acc_out, axis=0), prob),
        axis=0
    )
    max_diff = {
        "hessian_acc": abs_err(acc_out, pf_out2["accumulate_out"])
    }
    print(max_diff)
