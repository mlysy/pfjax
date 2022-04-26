import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import jax.tree_util as jtu
from functools import partial
import pfjax as pf
import pfjax.experimental.particle_filter as pfex
from pfjax.models import BMModel
# from pfjax.particle_filter import _lweight_to_prob


def expand_grid(**kwargs):
    """
    JAX equivalent of expand_grid in R.

    Unlike R, leftmost vectors are changing fastest.
    """
    keys = list(kwargs)
    out = jnp.meshgrid(*[kwargs[k] for k in keys])
    return {keys[i]: jnp.ravel(out[i]) for i in jnp.arange(len(out))}


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


def accumulate_brute(x_particles, ancestors, y_meas, theta, accumulator):
    """
    Brute force accumulator.

    Does everything except the final weighted average, as this needs to be done separately for the fisher information.
    """
    n_particles = x_particles.shape[1]
    x_particles_full = jax.vmap(
        lambda i: get_particles(i, x_particles, ancestors)
    )(jnp.arange(n_particles))
    x_particles_prev = x_particles_full[:, :-1]
    x_particles_curr = x_particles_full[:, 1:]
    y_curr = y_meas[1:]
    acc_out = jax.vmap(
        jax.vmap(
            accumulator,
            in_axes=(0, 0, 0, None)
        ),
        in_axes=(0, 0, None, None)
    )(x_particles_prev, x_particles_curr, y_curr, theta)
    acc_out = jtu.tree_map(lambda x: jnp.sum(x, axis=1), acc_out)
    return acc_out
    # return pfex._tree_mean(acc_out, logw)


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
    # Update: particle_filter_for no longer available here!
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
    pf_out2 = pfex.particle_accumulator(
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
    pf_out2 = pfex.particle_accumulator(
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
    pf_out1 = pfex.particle_accumulator(
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
    pf_out1 = pfex.particle_accumulator(
        bm_model, subkey, y_meas, theta, n_particles,
        history=True, accumulator=accumulate_score)

    # new pf without history
    pf_out2 = pfex.particle_accumulator(
        bm_model, subkey, y_meas, theta, n_particles,
        history=False, accumulator=accumulate_score)

    # brute force calculation
    x_particles = pf_out1["x_particles"]
    ancestors = pf_out1["resample_out"]["ancestors"]
    logw = pf_out1["logw"][n_obs-1]
    acc_out = accumulate_brute(
        x_particles=x_particles,
        ancestors=ancestors,
        y_meas=y_meas,
        theta=theta,
        accumulator=accumulate_score
    )
    acc_out = pfex._tree_mean(acc_out, logw)
    max_diff = {
        "score_acc": abs_err(acc_out, pf_out2["accumulate_out"])
    }
    print(max_diff)

if False:
    def accumulate_diff(x_prev, x_curr, y_curr, theta):
        r"""
        Accumulator for both score and hessian.
        """
        grad_meas = jax.grad(bm_model.meas_lpdf, argnums=2)
        grad_state = jax.grad(bm_model.state_lpdf, argnums=2)
        hess_meas = jax.jacfwd(jax.jacrev(bm_model.meas_lpdf, argnums=2),
                               argnums=2)
        hess_state = jax.jacfwd(jax.jacrev(bm_model.state_lpdf, argnums=2),
                                argnums=2)
        alpha = grad_meas(y_curr, x_curr, theta) + \
            grad_state(x_curr, x_prev, theta)
        beta = hess_meas(y_curr, x_curr, theta) + \
            hess_state(x_curr, x_prev, theta)
        return (alpha, beta)

    # new pf with history
    pf_out1 = pfex.particle_accumulator(
        bm_model, subkey, y_meas, theta, n_particles,
        history=True, accumulator=accumulate_diff)

    # new pf without history
    pf_out2 = pfex.particle_accumulator(
        bm_model, subkey, y_meas, theta, n_particles,
        history=False, accumulator=accumulate_diff)

    # brute force calculation
    x_particles = pf_out1["x_particles"]
    ancestors = pf_out1["resample_out"]["ancestors"]
    logw = pf_out1["logw"][n_obs-1]
    acc_out = accumulate_brute(
        x_particles=x_particles,
        ancestors=ancestors,
        y_meas=y_meas,
        theta=theta,
        accumulator=accumulate_diff
    )
    acc_out = pfex._tree_mean(acc_out, logw)
    max_diff = {
        "score_acc": abs_err(acc_out[0], pf_out2["accumulate_out"][0]),
        "hessian_acc": abs_err(acc_out[1], pf_out2["accumulate_out"][1])
    }
    print(max_diff)


# --- test auxillary_filter_linear ---------------------------------------------

if False:
    def accumulate_diff(x_prev, x_curr, y_curr, theta):
        r"""
        Accumulator for both score and hessian.
        """
        grad_meas = jax.grad(bm_model.meas_lpdf, argnums=2)
        grad_state = jax.grad(bm_model.state_lpdf, argnums=2)
        hess_meas = jax.jacfwd(jax.jacrev(bm_model.meas_lpdf, argnums=2),
                               argnums=2)
        hess_state = jax.jacfwd(jax.jacrev(bm_model.state_lpdf, argnums=2),
                                argnums=2)
        alpha = grad_meas(y_curr, x_curr, theta) + \
            grad_state(x_curr, x_prev, theta)
        beta = hess_meas(y_curr, x_curr, theta) + \
            hess_state(x_curr, x_prev, theta)
        return (alpha, beta)

    # accumulator with history
    pf_out1 = pfex.particle_accumulator(
        bm_model, subkey, y_meas, theta, n_particles,
        history=True,
        accumulator=accumulate_diff)
    # auxillary filter with history
    pf_out2 = pfex.auxillary_filter_linear(
        bm_model, subkey, y_meas, theta, n_particles,
        score=True, fisher=True,
        history=True)

    # check x_particles and logw
    max_diff = {k: abs_err(pf_out1[k],  pf_out2[k])
                for k in ["x_particles", "logw"]}
    print(max_diff)

    # # check ancestors
    # max_diff = {k: abs_err(pf_out1[k][n_obs-1], pf_out2["resample_out"][k])
    #             for k in ["ancestors"]}
    # print(max_diff)

    # check loglik
    max_diff = {
        "loglik": abs_err(pf_out1["loglik"], pf_out2["loglik"])
    }
    print(max_diff)

    # check score and fisher information
    x_particles = pf_out1["x_particles"]
    ancestors = pf_out1["resample_out"]["ancestors"]
    logw = pf_out1["logw"][n_obs-1]
    alpha, beta = accumulate_brute(
        x_particles=x_particles,
        ancestors=ancestors,
        y_meas=y_meas,
        theta=theta,
        accumulator=accumulate_diff
    )
    prob = pf.lwgt_to_prob(logw)
    score = jax.vmap(jnp.multiply)(prob, alpha)
    hess = jax.vmap(
        lambda p, a, b: p * (jnp.outer(a, a) + b)
    )(prob, alpha, beta)
    score, hess = jtu.tree_map(lambda x: jnp.sum(x, axis=0), (score, hess))
    max_diff = {
        "score": abs_err(score, pf_out2["score"]),
        "fisher": abs_err(hess - jnp.outer(score, score), pf_out2["fisher"])
    }
    print(max_diff)

# --- test auxillary_filter_quad -----------------------------------------------

if False:
    # test for vs vmap
    job_descr = expand_grid(
        history=jnp.array([False, True]),
        score=jnp.array([False, True]),
        fisher=jnp.array([False, True])
    )
    for i in jnp.arange(job_descr["history"].size):
        history = job_descr["history"][i]
        score = job_descr["score"][i]
        fisher = job_descr["fisher"][i]
        print("history = {}, score = {}, fisher = {}".format(
            history, score, fisher)
        )
        # auxillary filter for-loop
        pf_out1 = pfex.auxillary_filter_quad(
            bm_model, subkey, y_meas, theta, n_particles,
            score=score, fisher=fisher,
            history=history, tilde_for=True)
        # auxillary filter vmap
        pf_out2 = pfex.auxillary_filter_quad(
            bm_model, subkey, y_meas, theta, n_particles,
            score=score, fisher=fisher,
            history=history, tilde_for=False)
        max_diff = {k: abs_err(pf_out1[k], pf_out2[k])
                    for k in pf_out1.keys()}
        print(max_diff)


if True:
    # test history vs no history
    job_descr = expand_grid(
        score=jnp.array([False, True]),
        fisher=jnp.array([False, True])
    )
    for i in jnp.arange(job_descr["score"].size):
        score = job_descr["score"][i]
        fisher = job_descr["fisher"][i]
        print("score = {}, fisher = {}".format(
            score, fisher)
        )
        # auxillary filter no history
        pf_out1 = pfex.auxillary_filter_quad(
            bm_model, subkey, y_meas, theta, n_particles,
            score=score, fisher=fisher,
            history=False, tilde_for=False)
        # auxillary filter with history
        pf_out2 = pfex.auxillary_filter_quad(
            bm_model, subkey, y_meas, theta, n_particles,
            score=score, fisher=fisher,
            history=True, tilde_for=False)
        # check loglik, score, fisher
        keys = ["loglik"]
        keys = keys + ["score"] if score or fisher else keys
        keys = keys + ["fisher"] if fisher else keys
        max_diff = {k: abs_err(pf_out1[k], pf_out2[k]) for k in keys}
        print(max_diff)
