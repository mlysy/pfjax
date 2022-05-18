import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import jax.tree_util as jtu
from functools import partial
import pfjax as pf
import pfjax.experimental.particle_filter as pfex
import pfjax.tests as pftest
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


def rel_err(x1, x2):
    return jnp.max(jnp.abs(x1-x2) / (jnp.abs(x1+x2) + .1))


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

# --- basic particle filter ----------------------------------------------------

if False:
    # test against for-loop version
    # old pf with for-loop
    pf_out1 = pftest.particle_filter_for(
        bm_model, subkey, y_meas, theta, n_particles)

    job_descr = expand_grid(
        history=jnp.array([False, True]),
    )
    for i in jnp.arange(job_descr["history"].size):
        history = job_descr["history"][i]
        print("history = {}".format(history))
        # new pf
        pf_out2 = pfex.particle_filter(
            bm_model, subkey, y_meas, theta, n_particles,
            score=False, fisher=False, history=history)
        # check outputs
        if history:
            max_diff = {k: rel_err(pf_out1[k], pf_out2[k])
                        for k in ["x_particles", "logw"]}
            max_diff["ancestors"] = rel_err(
                x1=pf_out1["ancestors"],
                x2=pf_out2["resample_out"]["ancestors"]
            )
        else:
            max_diff = {k: rel_err(pf_out1[k][n_obs-1],  pf_out2[k])
                        for k in ["x_particles", "logw"]}
        max_diff["loglik"] = rel_err(pf.particle_loglik(pf_out1["logw"]),
                                     pf_out2["loglik"])
        print_dict(max_diff)


if False:
    # test online vs brute force derivatives
    # also test with and without history gives same results
    def accumulate_deriv(x_prev, x_curr, y_curr, theta):
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

    # pf with history, no derivatives
    pf_out1 = pfex.particle_filter(
        bm_model, subkey, y_meas, theta, n_particles,
        score=False, fisher=False, history=True)

    job_descr = expand_grid(
        history=jnp.array([True]),
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
        # pf various history/derivatives
        pf_out2 = pfex.particle_filter(
            bm_model, subkey, y_meas, theta, n_particles,
            score=score, fisher=fisher, history=history
        )
        # check outputs
        if history:
            max_diff = {k: rel_err(pf_out1[k],  pf_out2[k])
                        for k in ["x_particles", "logw"]}
            max_diff["ancestors"] = rel_err(
                x1=pf_out1["resample_out"]["ancestors"],
                x2=pf_out2["resample_out"]["ancestors"]
            )
        else:
            max_diff = {k: rel_err(pf_out1[k][n_obs-1],  pf_out2[k])
                        for k in ["x_particles", "logw"]}

        max_diff["loglik"] = rel_err(pf_out1["loglik"], pf_out2["loglik"])
        if score or fisher:
            # score and hess using smoothing accumulator
            x_particles = pf_out1["x_particles"]
            ancestors = pf_out1["resample_out"]["ancestors"]
            logw = pf_out1["logw"][n_obs-1]
            alpha, beta = pftest.accumulate_smooth(
                logw=logw,
                x_particles=x_particles,
                ancestors=ancestors,
                y_meas=y_meas,
                theta=theta,
                accumulator=accumulate_deriv,
                mean=False
            )
            prob = pf.lwgt_to_prob(logw)
            _score = jax.vmap(jnp.multiply)(prob, alpha)
            _hess = jax.vmap(
                lambda p, a, b: p * (jnp.outer(a, a) + b)
            )(prob, alpha, beta)
            _score, _hess = jtu.tree_map(
                lambda x: jnp.sum(x, axis=0), (_score, _hess))
            max_diff["score"] = rel_err(_score, pf_out2["score"])
            if fisher:
                max_diff["fisher"] = rel_err(
                    _hess - jnp.outer(_score, _score),
                    pf_out2["fisher"]
                )
        print_dict(max_diff)


# --- test rao-blackwellized particle filter -----------------------------------


if False:
    # test for-loop vs vmap
    # for-loop is very slow, so skip history=True
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
        # rb filter for-loop
        pf_out1 = pfex.particle_filter_rb(
            bm_model, subkey, y_meas, theta, n_particles,
            score=score, fisher=fisher,
            history=history, tilde_for=True)
        # rb filter vmap
        pf_out2 = pfex.particle_filter_rb(
            bm_model, subkey, y_meas, theta, n_particles,
            score=score, fisher=fisher,
            history=history, tilde_for=False)
        # max_diff = {k: rel_err(pf_out1[k], pf_out2[k])
        #             for k in pf_out1.keys()}
        max_diff = jtu.tree_map(rel_err, pf_out1, pf_out2)
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
        pf_out1 = pfex.particle_filter_rb(
            bm_model, subkey, y_meas, theta, n_particles,
            score=score, fisher=fisher,
            history=False, tilde_for=False)
        # auxillary filter with history
        pf_out2 = pfex.particle_filter_rb(
            bm_model, subkey, y_meas, theta, n_particles,
            score=score, fisher=fisher,
            history=True, tilde_for=False)
        # check loglik, score, fisher
        keys = ["loglik"]
        keys = keys + ["score"] if score or fisher else keys
        keys = keys + ["fisher"] if fisher else keys
        max_diff = {k: rel_err(pf_out1[k], pf_out2[k]) for k in keys}
        print_dict(max_diff)


if False:
    # test online vs brute force derivative calculations
    # gradient and hessian functions
    grad_meas = jax.grad(bm_model.meas_lpdf, argnums=2)
    grad_state = jax.grad(bm_model.state_lpdf, argnums=2)
    hess_meas = jax.jacfwd(jax.jacrev(bm_model.meas_lpdf, argnums=2),
                           argnums=2)
    hess_state = jax.jacfwd(jax.jacrev(bm_model.state_lpdf, argnums=2),
                            argnums=2)

    def grad_step(x_curr, x_prev, y_curr,
                  logw_prev, logw_aux, alpha_prev, beta_prev):
        """
        Update logw_targ, alpha, and beta.
        """
        logw_targ = bm_model.meas_lpdf(
            y_curr=y_curr,
            x_curr=x_curr,
            theta=theta
        ) + bm_model.state_lpdf(
            x_curr=x_curr,
            x_prev=x_prev,
            theta=theta
        ) + logw_prev
        logw_prop = bm_model.prop_lpdf(
            x_curr=x_curr,
            x_prev=x_prev,
            y_curr=y_curr,
            theta=theta
        ) + logw_aux
        alpha = grad_state(x_curr, x_prev, theta) + \
            grad_meas(y_curr, x_curr, theta) + \
            alpha_prev
        beta = jnp.outer(alpha, alpha) + \
            hess_meas(y_curr, x_curr, theta) + \
            hess_state(x_curr, x_prev, theta) + \
            beta_prev
        return {
            "logw_targ": logw_targ,
            "logw_prop": logw_prop,
            "alpha": alpha,
            "beta": beta
        }

    job_descr = expand_grid(
        history=jnp.array([False, True]),
        score=jnp.array([False, True]),
        fisher=jnp.array([False, True])
    )

    # auxillary filter vmap
    pf_out1 = pfex.particle_filter_rb(
        model=bm_model,
        key=subkey,
        y_meas=y_meas,
        theta=theta,
        n_particles=n_particles,
        score=False,
        fisher=False,
        history=True, tilde_for=False
    )
    # initialize
    n_theta = theta.size
    logw_prev = pf_out1["logw_bar"][0]
    alpha_prev = jnp.zeros((n_particles, n_theta))
    beta_prev = jnp.zeros((n_particles, n_theta, n_theta))
    loglik2 = jsp.special.logsumexp(logw_prev)
    # update for every observation
    for i_curr in range(1, n_obs):
        x_prev = pf_out1["x_particles"][i_curr-1]
        x_curr = pf_out1["x_particles"][i_curr]
        y_curr = y_meas[i_curr]
        logw_aux = logw_prev
        # manual update calculation
        grad_full = jax.vmap(
            jax.vmap(
                grad_step,
                in_axes=(None, 0, None, 0, 0, 0, 0)
            ),
            in_axes=(0, None, None, None, None, None, None)
        )(x_curr, x_prev, y_curr, logw_prev, logw_aux,
          alpha_prev, beta_prev)
        logw_curr = jax.vmap(
            lambda ltarg, lprop:
            jsp.special.logsumexp(ltarg) - jsp.special.logsumexp(lprop)
        )(grad_full["logw_targ"], grad_full["logw_prop"])
        loglik2 = loglik2 + jsp.special.logsumexp(logw_curr)
        alpha_curr = jax.vmap(
            pf.utils.tree_mean
        )(grad_full["alpha"], grad_full["logw_targ"])
        beta_curr = jax.vmap(
            pf.utils.tree_mean
        )(grad_full["beta"], grad_full["logw_targ"]) - \
            jax.vmap(
            jnp.outer
        )(alpha_curr, alpha_curr)
        # set prev to curr
        logw_prev = logw_curr
        alpha_prev = alpha_curr
        beta_prev = beta_curr
    # finalize calculations
    loglik2 = loglik2 - n_obs * jnp.log(n_particles)
    gamma_curr = jax.vmap(
        lambda a, b: jnp.outer(a, a) + b
    )(alpha_curr, beta_curr)
    score2 = pf.utils.tree_mean(alpha_curr, logw_curr)
    fisher2 = pf.utils.tree_mean(gamma_curr, logw_curr) - \
        jnp.outer(score2, score2)
    # check against different versions
    for i in jnp.arange(job_descr["history"].size):
        history = job_descr["history"][i]
        score = job_descr["score"][i]
        fisher = job_descr["fisher"][i]
        print("history = {}, score = {}, fisher = {}".format(
            history, score, fisher)
        )
        # correspoinding particle filter
        pf_out2 = pfex.particle_filter_rb(
            model=bm_model,
            key=subkey,
            y_meas=y_meas,
            theta=theta,
            n_particles=n_particles,
            score=score,
            fisher=fisher,
            history=history,
            tilde_for=False
        )
        # calculate difference
        max_diff = {"loglik": rel_err(pf_out2["loglik"], loglik2)}
        if score or fisher:
            max_diff["score"] = rel_err(pf_out2["score"], score2)
        if fisher:
            max_diff["fisher"] = rel_err(pf_out2["fisher"], fisher2)
        print_dict(max_diff)


# --- scratch ------------------------------------------------------------------

# don't bother with anything below here

# --- test for-loop ------------------------------------------------------------

if False:
    # pf with for-loop
    pf_out1 = pftest.particle_filter_for(
        bm_model, subkey, y_meas, theta, n_particles)
    # pf without for-loop
    pf_out2 = pf.particle_filter(
        bm_model, subkey, y_meas, theta, n_particles)

    max_diff = {
        k: jnp.max(jnp.abs(pf_out1[k] - pf_out2[k]))
        for k in pf_out1.keys()
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
        "ancestors_acc": rel_err(ancestors1, ancestors2)
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
        "score_acc": rel_err(acc_out, pf_out2["accumulate_out"])
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
        "score_acc": rel_err(acc_out[0], pf_out2["accumulate_out"][0]),
        "hessian_acc": rel_err(acc_out[1], pf_out2["accumulate_out"][1])
    }
    print_dict(max_diff)


if False:
    def accumulate_deriv(x_prev, x_curr, y_curr, theta):
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
        pf_out1 = pfex.particle_filter(
            bm_model, subkey, y_meas, theta, n_particles,
            score=False, fisher=False, history=True)
        # auxillary filter with history
        pf_out2 = pfex.auxillary_filter_linear(
            bm_model, subkey, y_meas, theta, n_particles,
            score=score, fisher=fisher,
            history=history)

        # check x_particles and logw
        max_diff = {k: rel_err(pf_out1[k],  pf_out2[k])
                    for k in ["x_particles", "logw"]}
        print_dict(max_diff)

        # check ancestors
        max_diff = {k: rel_err(pf_out1["resample_out"][k], pf_out2[k])
                    for k in ["ancestors"]}
        print_dict(max_diff)

        # check loglik
        max_diff = {
            "loglik": rel_err(pf_out1["loglik"], pf_out2["loglik"])
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
            max_diff["score"] = rel_err(_score, pf_out2["score"])
            if fisher:
                max_diff["fisher"] = rel_err(
                    _hess - jnp.outer(_score, _score),
                    pf_out2["fisher"]
                )

        print_dict(max_diff)

if False:
    # test linear vs quad
    # update: no longer relevant since underlying PF is now different
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
        max_diff = {k: rel_err(pf_out1[k], pf_out2[k])
                    for k in ["loglik", "x_particles"]}
        print_dict(max_diff)