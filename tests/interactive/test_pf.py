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


def print_dict(x):
    [print('{} : {}'.format(key, value)) for key, value in x.items()]


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
    print_dict(max_diff)

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
    print_dict(max_diff)

    # check ancestors
    max_diff = {k: abs_err(pf_out1[k], pf_out2["resample_out"][k])
                for k in ["ancestors"]}
    print_dict(max_diff)

    # check loglik
    max_diff = {
        "loglik": abs_err(pf.particle_loglik(pf_out1["logw"]),
                          pf_out2["loglik"])
    }
    print_dict(max_diff)

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
    print_dict(max_diff)

    # check ancestors
    max_diff = {k: abs_err(pf_out1[k][n_obs-1], pf_out2["resample_out"][k])
                for k in ["ancestors"]}
    print_dict(max_diff)

    # check loglik
    max_diff = {
        "loglik": abs_err(pf.particle_loglik(pf_out1["logw"]),
                          pf_out2["loglik"])
    }
    print_dict(max_diff)


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
    print_dict(max_diff)


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
    acc_out = pfex.accumulate_smooth(
        logw=logw,
        x_particles=x_particles,
        ancestors=ancestors,
        y_meas=y_meas,
        theta=theta,
        accumulator=accumulate_score
    )
    max_diff = {
        "score_acc": abs_err(acc_out, pf_out2["accumulate_out"])
    }
    print_dict(max_diff)

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
    acc_out = pfex.accumulate_smooth(
        logw=logw,
        x_particles=x_particles,
        ancestors=ancestors,
        y_meas=y_meas,
        theta=theta,
        accumulator=accumulate_diff
    )
    max_diff = {
        "score_acc": abs_err(acc_out[0], pf_out2["accumulate_out"][0]),
        "hessian_acc": abs_err(acc_out[1], pf_out2["accumulate_out"][1])
    }
    print_dict(max_diff)


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

    job_descr = expand_grid(
        history=jnp.array([True]),  # False doesn't calculate the right thing

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
        # accumulator with history
        pf_out1 = pfex.particle_accumulator(
            bm_model, subkey, y_meas, theta, n_particles,
            history=history,
            accumulator=accumulate_diff)
        # auxillary filter with history
        pf_out2 = pfex.auxillary_filter_linear(
            bm_model, subkey, y_meas, theta, n_particles,
            score=score, fisher=fisher,
            history=history)

        # check x_particles and logw
        max_diff = {k: abs_err(pf_out1[k],  pf_out2[k])
                    for k in ["x_particles", "logw"]}
        print_dict(max_diff)

        # check ancestors
        max_diff = {k: abs_err(pf_out1["resample_out"][k], pf_out2[k])
                    for k in ["ancestors"]}
        print_dict(max_diff)

        # check loglik
        max_diff = {
            "loglik": abs_err(pf_out1["loglik"], pf_out2["loglik"])
        }

        if score or fisher:
            # score and hess using smoothing accumulator
            if history:
                x_particles = pf_out1["x_particles"]
                ancestors = pf_out1["resample_out"]["ancestors"]
                logw = pf_out1["logw"][n_obs-1]
                alpha, beta = pfex.accumulate_smooth(
                    logw=logw,
                    x_particles=x_particles,
                    ancestors=ancestors,
                    y_meas=y_meas,
                    theta=theta,
                    accumulator=accumulate_diff,
                    mean=False
                )
                prob = pf.lwgt_to_prob(logw)
                _score = jax.vmap(jnp.multiply)(prob, alpha)
                _hess = jax.vmap(
                    lambda p, a, b: p * (jnp.outer(a, a) + b)
                )(prob, alpha, beta)
                _score, _hess = jtu.tree_map(
                    lambda x: jnp.sum(x, axis=0), (_score, _hess))
            else:
                # score and hess using filtering accumulator
                _score, _hess = pf_out1["accumulate_out"]
            max_diff["score"] = abs_err(_score, pf_out2["score"])
            if fisher:
                max_diff["fisher"] = abs_err(
                    _hess - jnp.outer(_score, _score),
                    pf_out2["fisher"]
                )

        print_dict(max_diff)

# --- test auxillary_filter_quad -----------------------------------------------

if False:
    # test linear vs quad
    job_descr = expand_grid(
        history=jnp.array([False, True])
    )
    for i in jnp.arange(job_descr["history"].size):
        history = job_descr["history"][i]
        print("history = {}".format(history))
        # auxillary filter linear
        pf_out1 = pfex.auxillary_filter_linear(
            bm_model, subkey, y_meas, theta, n_particles,
            score=False, fisher=False,
            history=history)
        # auxillary filter quadratic
        pf_out2 = pfex.auxillary_filter_quad(
            bm_model, subkey, y_meas, theta, n_particles,
            score=False, fisher=False,
            history=history, tilde_for=False)
        max_diff = {k: abs_err(pf_out1[k], pf_out2[k])
                    for k in ["loglik", "x_particles"]}
        print_dict(max_diff)

if False:
    # test for vs vmap
    job_descr = expand_grid(
        history=jnp.array([False]),
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
        print_dict(max_diff)


if False:
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
        print_dict(max_diff)

pf_out1 = pfex.auxillary_filter_quad(
    model=bm_model,
    key=subkey,
    y_meas=y_meas[0:2],
    theta=theta,
    n_particles=100,
    score=True,
    fisher=True,
    history=True,
    tilde_for=False
)

# brute force calculation of score accumulator
grad_meas = jax.grad(bm_model.meas_lpdf, argnums=2)
grad_state = jax.grad(bm_model.state_lpdf, argnums=2)


def alpha_fun(x_curr, x_prev, y_curr, alpha_prev, logw_prev):
    return {
        "logf": bm_model.state_lpdf(
            x_curr=x_curr,
            x_prev=x_prev,
            theta=theta
        ) + logw_prev,
        "score": grad_state(
            x_curr,
            x_prev,
            theta
        ) + grad_meas(
            y_curr,
            x_curr,
            theta
        ) + alpha_prev
    }


x_curr = pf_out1["x_particles"][1]
x_prev = pf_out1["x_particles"][0]
y_curr = y_meas[1]
logw_prev = pf_out1["logw_bar"][0]
logw_curr = pf_out1["logw_bar"][1]
alpha_prev = jnp.zeros((100,))

alpha_full = jax.vmap(
    jax.vmap(
        alpha_fun,
        in_axes=(None, 0, None, 0, 0)
    ),
    in_axes=(0, None, None, None, None)
)(x_curr, x_prev, y_curr, alpha_prev, logw_prev)

i_curr = 2
i_prev = 13
alpha_fun(x_curr[i_curr], x_prev[i_prev], y_curr,
          alpha_prev[i_prev], logw_prev[i_prev])
(alpha_full["logf"][i_curr, i_prev], alpha_full["score"][i_curr, i_prev])

alpha_curr = jax.vmap(
    pfex._tree_mean
)(alpha_full["score"], alpha_full["logf"])

score = pfex._tree_mean(alpha_curr, logw_curr)


# true gradient value
jax.grad(bm_model.loglik_exact, argnums=1)(y_meas[0:2], theta)

# filter approximation
score
pf_out1["score"]

# true hessian value
jax.jacfwd(jax.jacrev(bm_model.loglik_exact,
                      argnums=1), argnums=1)(y_meas[0:2], theta)
