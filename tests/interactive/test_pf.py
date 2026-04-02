import jax
import jax.numpy as jnp
# import jax.scipy as jsp
# import jax.random as random
# import jax.tree_util as jtu
# from functools import partial
# import pfjax.experimental.particle_filter as pfex
# import pfjax.tests as pftest
# from pfjax.models import BMModel
# from pfjax.particle_filter import _lweight_to_prob
import pfjax as pf
import tests.utils
from jax.scipy.special import logsumexp
from pfjax.models import BMModel
from tests.interactive.basic_filter import BasicFilter
from tests.ss_model import SSModel
from tests.utils import expand_grid, rel_err

jax.config.update("jax_enable_x64", True)


# --- test stop_gradient -------------------------------------------------------


class BMModelSimple(pf.experimental.base_model.BaseModel):
    def __init__(self):
        """
        A simplified version of BM Model:

        x_0 ~ Lebesgue
        x_t ~  + Normal(beta * x_{t-1}, sigma^2)
        y_t ~ x_t + Normal(0, 1)
        """
        super().__init__(bootstrap=False)

    def state_lpdf(self, x_curr, x_prev, theta):
        beta, sigma = theta
        return jax.scipy.stats.norm.logpdf(
            x=x_curr,
            loc=beta * x_prev,
            scale=sigma,
        )

    def state_sample(self, key, x_prev, theta):
        beta, sigma = theta
        return beta * x_prev + sigma * jax.random.normal(key)

    def meas_lpdf(self, y_curr, x_curr, theta):
        return jax.scipy.stats.norm.logpdf(
            x=y_curr,
            loc=x_curr,
            scale=1.0,
        )

    def meas_sample(self, key, x_curr, theta):
        return x_curr + jax.random.normal(key)

    def pf_init(self, key, y_init, theta):
        return (y_init + jax.random.normal(key), 0.0)

    def pf_step(self, key, x_prev, y_curr, theta):
        x_curr = y_curr + jax.random.normal(key)
        lp_prop = jax.scipy.stats.norm.logpdf(x=x_curr, loc=0.0, scale=1.0)
        lp_targ = self.state_lpdf(x_curr=x_curr, x_prev=x_prev, theta=theta)
        lp_targ = lp_targ + self.meas_lpdf(y_curr=y_curr, x_curr=x_curr, theta=theta)
        return (x_curr, lp_targ - lp_prop)


key = jax.random.PRNGKey(0)
# parameter values
beta = 1.3
sigma = 5.1
theta = jnp.array([beta, sigma])
# data specification
n_obs = 5
x_init = jnp.array(0.0)
bm_model = BMModelSimple()
# simulate without for-loop
y_meas, x_state = pf.simulate(bm_model, key, n_obs, x_init, theta)

# particle filter specification
n_particles = 1
key, subkey = jax.random.split(key)

n_obs_test = 3
out1 = pf.particle_filter(
    model=bm_model,
    key=key,
    y_meas=y_meas[0:n_obs_test],
    theta=theta,
    n_particles=n_particles,
    score=True,
)

bm_filter = BasicFilter(model=bm_model)
out2 = bm_filter(
    key,
    y_meas[0:n_obs_test],
    theta,
    n_particles,
)
out3 = jax.grad(bm_filter, argnums=2, has_aux=True)(
    key,
    y_meas[0:n_obs_test],
    theta,
    n_particles,
)

print(f'out1_loglik={out1["loglik"]}, out2_loglik={out2[0]}')

print(f'out1_score={out1["score"]}, out3_score={out3[0]}')

# --- ss_model -----------------------------------------------------------------

ss_inputs = tests.utils.ss_setup()

ss_model = ss_inputs["model"](**ss_inputs["model_args"])
jax.grad(ss_model.meas_lpdf, argnums=2)(
    jnp.zeros((2,)), jnp.zeros((3,)), ss_inputs["theta"]
)

tests.utils.test_particle_filter_deriv(**ss_inputs)

# --- test dice ----------------------------------------------------------------


def prop_sim(key, theta):
    return theta + jnp.exp(theta) * jax.random.normal(key)


def prop_lpdf(x, theta):
    return jax.scipy.stats.norm.logpdf(x, loc=theta, scale=jnp.exp(theta))


def targ_lpdf(x, theta):
    return jax.scipy.stats.t.logpdf(x, df=10, loc=theta, scale=jnp.exp(theta))


def f(x, theta):
    return (theta + x) ** 2


def is_est(key, n_particles, theta):
    keys = jax.random.split(key, num=n_particles)
    x_sim = jax.vmap(prop_sim, in_axes=(0, None))(jnp.array(keys), theta)
    logw_targ = jax.vmap(targ_lpdf, in_axes=(0, None))(x_sim, theta)
    logw_prop = jax.vmap(prop_lpdf, in_axes=(0, None))(x_sim, theta)
    logw = logw_targ - logw_prop
    prob = pf.utils.logw_to_prob(logw)
    f_sim = jax.vmap(f, in_axes=(0, None))(x_sim, theta)
    return jnp.sum(prob * f_sim)


def is_score(key, n_particles, theta):
    def g(x, theta):
        ans = f(x, theta) * jax.grad(targ_lpdf, argnums=1)(x, theta)
        return ans + jax.grad(f, argnums=1)(x, theta)

    keys = jax.random.split(key, num=n_particles)
    x_sim = jax.vmap(prop_sim, in_axes=(0, None))(jnp.array(keys), theta)
    logw_targ = jax.vmap(targ_lpdf, in_axes=(0, None))(x_sim, theta)
    logw_prop = jax.vmap(prop_lpdf, in_axes=(0, None))(x_sim, theta)
    logw = logw_targ - logw_prop
    prob = pf.utils.logw_to_prob(logw)
    g_sim = jax.vmap(g, in_axes=(0, None))(x_sim, theta)
    return jnp.sum(prob * g_sim)


def is_dice(key, n_particles, theta):
    keys = jax.random.split(key, num=n_particles)
    x_sim = jax.vmap(prop_sim, in_axes=(0, None))(jnp.array(keys), theta)
    x_sim = jax.lax.stop_gradient(x_sim)
    logw_targ = jax.vmap(targ_lpdf, in_axes=(0, None))(x_sim, theta)
    logw_prop = jax.vmap(prop_lpdf, in_axes=(0, None))(x_sim, theta)
    logw = logw_targ - logw_prop
    prob = pf.utils.logw_to_prob(logw)
    f_sim = jax.vmap(f, in_axes=(0, None))(x_sim, theta)
    f_sim = f_sim * jnp.exp(logw_targ - jax.lax.stop_gradient(logw_targ))
    return jnp.sum(prob * f_sim)


theta = 1.0
n_particles = 10
key = jax.random.PRNGKey(0)

is_est(key, n_particles=n_particles, theta=theta)
is_score(key, n_particles=n_particles, theta=theta)

is_dice(key, n_particles=n_particles, theta=theta)
jax.grad(is_dice, argnums=2)(key, n_particles, theta)

# --- bm_model -----------------------------------------------------------------

key = jax.random.PRNGKey(0)
# parameter values
mu = 5
sigma = 1
tau = 0.1
theta = jnp.array([mu, sigma, tau])
# data specification
dt = 0.1
n_obs = 5
x_init = jnp.array(0.0)
bm_model = BMModel(dt=dt)
# simulate without for-loop
y_meas, x_state = pf.simulate(bm_model, key, n_obs, x_init, theta)

# particle filter specification
n_particles = 7
key, subkey = jax.random.split(key)

out1 = pf.particle_filter(
    model=bm_model,
    key=key,
    y_meas=y_meas[0:2],
    theta=theta,
    n_particles=n_particles,
    score=True,
)

bm_filter = BasicFilter(model=bm_model)
out2 = bm_filter(
    key,
    y_meas[0:2],
    theta,
    n_particles,
)
out3 = jax.grad(bm_filter, argnums=2, has_aux=True)(
    key,
    y_meas[0:2],
    theta,
    n_particles,
)

# --- basic particle filter ----------------------------------------------------

# def expand_grid(**kwargs):
#     """
#     JAX equivalent of expand_grid in R.

#     Unlike R, leftmost vectors are changing fastest.
#     """
#     keys = list(kwargs)
#     out = jnp.meshgrid(*[kwargs[k] for k in keys])
#     return {keys[i]: jnp.ravel(out[i]) for i in jnp.arange(len(out))}


# def rel_err(x1, x2):
#     return jnp.max(jnp.abs(x1 - x2) / (jnp.abs(x1 + x2) + 0.1))


def print_dict(x):
    [print("{} : {}".format(key, value)) for key, value in x.items()]


if False:
    # test against for-loop version
    # old pf with for-loop
    pf_out1 = pftest.particle_filter_for(bm_model, subkey, y_meas, theta, n_particles)

    job_descr = expand_grid(
        history=jnp.array([False, True]),
    )
    for i in jnp.arange(job_descr["history"].size):
        history = job_descr["history"][i]
        print("history = {}".format(history))
        # new pf
        pf_out2 = pfex.particle_filter(
            bm_model,
            subkey,
            y_meas,
            theta,
            n_particles,
            score=False,
            fisher=False,
            history=history,
        )
        # check outputs
        if history:
            max_diff = {
                k: rel_err(pf_out1[k], pf_out2[k]) for k in ["x_particles", "logw"]
            }
            max_diff["ancestors"] = rel_err(
                x1=pf_out1["ancestors"], x2=pf_out2["resample_out"]["ancestors"]
            )
        else:
            max_diff = {
                k: rel_err(pf_out1[k][n_obs - 1], pf_out2[k])
                for k in ["x_particles", "logw"]
            }
        max_diff["loglik"] = rel_err(
            pf.particle_loglik(pf_out1["logw"]), pf_out2["loglik"]
        )
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
        hess_meas = jax.jacfwd(jax.jacrev(bm_model.meas_lpdf, argnums=2), argnums=2)
        hess_state = jax.jacfwd(jax.jacrev(bm_model.state_lpdf, argnums=2), argnums=2)
        alpha = grad_meas(y_curr, x_curr, theta) + grad_state(x_curr, x_prev, theta)
        beta = hess_meas(y_curr, x_curr, theta) + hess_state(x_curr, x_prev, theta)
        return (alpha, beta)

    # pf with history, no derivatives
    pf_out1 = pfex.particle_filter(
        bm_model,
        subkey,
        y_meas,
        theta,
        n_particles,
        score=False,
        fisher=False,
        history=True,
    )

    job_descr = expand_grid(
        history=jnp.array([True]),
        score=jnp.array([False, True]),
        fisher=jnp.array([False, True]),
    )

    for i in jnp.arange(job_descr["history"].size):
        history = job_descr["history"][i]
        score = job_descr["score"][i]
        fisher = job_descr["fisher"][i]
        print("history = {}, score = {}, fisher = {}".format(history, score, fisher))
        # pf various history/derivatives
        pf_out2 = pfex.particle_filter(
            bm_model,
            subkey,
            y_meas,
            theta,
            n_particles,
            score=score,
            fisher=fisher,
            history=history,
        )
        # check outputs
        if history:
            max_diff = {
                k: rel_err(pf_out1[k], pf_out2[k]) for k in ["x_particles", "logw"]
            }
            max_diff["ancestors"] = rel_err(
                x1=pf_out1["resample_out"]["ancestors"],
                x2=pf_out2["resample_out"]["ancestors"],
            )
        else:
            max_diff = {
                k: rel_err(pf_out1[k][n_obs - 1], pf_out2[k])
                for k in ["x_particles", "logw"]
            }

        max_diff["loglik"] = rel_err(pf_out1["loglik"], pf_out2["loglik"])
        if score or fisher:
            # score and hess using smoothing accumulator
            x_particles = pf_out1["x_particles"]
            ancestors = pf_out1["resample_out"]["ancestors"]
            logw = pf_out1["logw"][n_obs - 1]
            alpha, beta = pftest.accumulate_smooth(
                logw=logw,
                x_particles=x_particles,
                ancestors=ancestors,
                y_meas=y_meas,
                theta=theta,
                accumulator=accumulate_deriv,
                mean=False,
            )
            prob = pf.logw_to_prob(logw)
            _score = jax.vmap(jnp.multiply)(prob, alpha)
            _hess = jax.vmap(lambda p, a, b: p * (jnp.outer(a, a) + b))(
                prob, alpha, beta
            )
            _score, _hess = jtu.tree_map(lambda x: jnp.sum(x, axis=0), (_score, _hess))
            max_diff["score"] = rel_err(_score, pf_out2["score"])
            if fisher:
                max_diff["fisher"] = rel_err(
                    _hess - jnp.outer(_score, _score), pf_out2["fisher"]
                )
        print_dict(max_diff)


# --- test rao-blackwellized particle filter -----------------------------------


if False:
    # test for-loop vs vmap
    # for-loop is very slow, so skip history=True
    job_descr = expand_grid(
        history=jnp.array([False]),
        score=jnp.array([False, True]),
        fisher=jnp.array([False, True]),
    )
    for i in jnp.arange(job_descr["history"].size):
        history = job_descr["history"][i]
        score = job_descr["score"][i]
        fisher = job_descr["fisher"][i]
        print("history = {}, score = {}, fisher = {}".format(history, score, fisher))
        # rb filter for-loop
        pf_out1 = pfex.particle_filter_rb(
            bm_model,
            subkey,
            y_meas,
            theta,
            n_particles,
            score=score,
            fisher=fisher,
            history=history,
            tilde_for=True,
        )
        # rb filter vmap
        pf_out2 = pfex.particle_filter_rb(
            bm_model,
            subkey,
            y_meas,
            theta,
            n_particles,
            score=score,
            fisher=fisher,
            history=history,
            tilde_for=False,
        )
        # max_diff = {k: rel_err(pf_out1[k], pf_out2[k])
        #             for k in pf_out1.keys()}
        max_diff = jtu.tree_map(rel_err, pf_out1, pf_out2)
        print_dict(max_diff)


if False:
    # test history vs no history
    job_descr = expand_grid(
        score=jnp.array([False, True]), fisher=jnp.array([False, True])
    )
    for i in jnp.arange(job_descr["score"].size):
        score = job_descr["score"][i]
        fisher = job_descr["fisher"][i]
        print("score = {}, fisher = {}".format(score, fisher))
        # auxillary filter no history
        pf_out1 = pfex.particle_filter_rb(
            bm_model,
            subkey,
            y_meas,
            theta,
            n_particles,
            score=score,
            fisher=fisher,
            history=False,
            tilde_for=False,
        )
        # auxillary filter with history
        pf_out2 = pfex.particle_filter_rb(
            bm_model,
            subkey,
            y_meas,
            theta,
            n_particles,
            score=score,
            fisher=fisher,
            history=True,
            tilde_for=False,
        )
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
    hess_meas = jax.jacfwd(jax.jacrev(bm_model.meas_lpdf, argnums=2), argnums=2)
    hess_state = jax.jacfwd(jax.jacrev(bm_model.state_lpdf, argnums=2), argnums=2)

    def grad_step(x_curr, x_prev, y_curr, logw_prev, logw_aux, alpha_prev, beta_prev):
        """
        Update logw_targ, alpha, and beta.
        """
        logw_targ = (
            bm_model.meas_lpdf(y_curr=y_curr, x_curr=x_curr, theta=theta)
            + bm_model.state_lpdf(x_curr=x_curr, x_prev=x_prev, theta=theta)
            + logw_prev
        )
        logw_prop = (
            bm_model.prop_lpdf(x_curr=x_curr, x_prev=x_prev, y_curr=y_curr, theta=theta)
            + logw_aux
        )
        alpha = (
            grad_state(x_curr, x_prev, theta)
            + grad_meas(y_curr, x_curr, theta)
            + alpha_prev
        )
        beta = (
            jnp.outer(alpha, alpha)
            + hess_meas(y_curr, x_curr, theta)
            + hess_state(x_curr, x_prev, theta)
            + beta_prev
        )
        return {
            "logw_targ": logw_targ,
            "logw_prop": logw_prop,
            "alpha": alpha,
            "beta": beta,
        }

    job_descr = expand_grid(
        history=jnp.array([False, True]),
        score=jnp.array([False, True]),
        fisher=jnp.array([False, True]),
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
        history=True,
        tilde_for=False,
    )
    # initialize
    n_theta = theta.size
    logw_prev = pf_out1["logw_bar"][0]
    alpha_prev = jnp.zeros((n_particles, n_theta))
    beta_prev = jnp.zeros((n_particles, n_theta, n_theta))
    loglik2 = jsp.special.logsumexp(logw_prev)
    # update for every observation
    for i_curr in range(1, n_obs):
        x_prev = pf_out1["x_particles"][i_curr - 1]
        x_curr = pf_out1["x_particles"][i_curr]
        y_curr = y_meas[i_curr]
        logw_aux = logw_prev
        # manual update calculation
        grad_full = jax.vmap(
            jax.vmap(grad_step, in_axes=(None, 0, None, 0, 0, 0, 0)),
            in_axes=(0, None, None, None, None, None, None),
        )(x_curr, x_prev, y_curr, logw_prev, logw_aux, alpha_prev, beta_prev)
        logw_curr = jax.vmap(
            lambda ltarg, lprop: jsp.special.logsumexp(ltarg)
            - jsp.special.logsumexp(lprop)
        )(grad_full["logw_targ"], grad_full["logw_prop"])
        loglik2 = loglik2 + jsp.special.logsumexp(logw_curr)
        alpha_curr = jax.vmap(pf.utils.tree_mean)(
            grad_full["alpha"], grad_full["logw_targ"]
        )
        beta_curr = jax.vmap(pf.utils.tree_mean)(
            grad_full["beta"], grad_full["logw_targ"]
        ) - jax.vmap(jnp.outer)(alpha_curr, alpha_curr)
        # set prev to curr
        logw_prev = logw_curr
        alpha_prev = alpha_curr
        beta_prev = beta_curr
    # finalize calculations
    loglik2 = loglik2 - n_obs * jnp.log(n_particles)
    gamma_curr = jax.vmap(lambda a, b: jnp.outer(a, a) + b)(alpha_curr, beta_curr)
    score2 = pf.utils.tree_mean(alpha_curr, logw_curr)
    fisher2 = pf.utils.tree_mean(gamma_curr, logw_curr) - jnp.outer(score2, score2)
    # check against different versions
    for i in jnp.arange(job_descr["history"].size):
        history = job_descr["history"][i]
        score = job_descr["score"][i]
        fisher = job_descr["fisher"][i]
        print("history = {}, score = {}, fisher = {}".format(history, score, fisher))
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
            tilde_for=False,
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
    pf_out1 = pftest.particle_filter_for(bm_model, subkey, y_meas, theta, n_particles)
    # pf without for-loop
    pf_out2 = pf.particle_filter(bm_model, subkey, y_meas, theta, n_particles)

    max_diff = {k: jnp.max(jnp.abs(pf_out1[k] - pf_out2[k])) for k in pf_out1.keys()}
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
        bm_model,
        subkey,
        y_meas,
        theta,
        n_particles,
        history=True,
        accumulator=accumulate_ancestors,
    )

    # check ancestors
    ancestors1 = []
    ancestors2 = []
    for i in range(n_obs - 1):
        ancestors1 += [
            pf_out1["x_particles"][i, pf_out1["resample_out"]["ancestors"][i]]
        ]
        ancestors2 += [pf_out1["accumulate_out"][0][i]]
    ancestors1 = jnp.array(ancestors1)
    ancestors2 = jnp.array(ancestors2)
    max_diff = {"ancestors_acc": rel_err(ancestors1, ancestors2)}
    print_dict(max_diff)


if False:

    def accumulate_score(x_prev, x_curr, y_curr, theta):
        r"""
        Accumulator for score function.
        """
        grad_meas = jax.grad(bm_model.meas_lpdf, argnums=2)
        grad_state = jax.grad(bm_model.state_lpdf, argnums=2)
        return grad_meas(y_curr, x_curr, theta) + grad_state(x_curr, x_prev, theta)

    # new pf with history
    pf_out1 = pfex.particle_accumulator(
        bm_model,
        subkey,
        y_meas,
        theta,
        n_particles,
        history=True,
        accumulator=accumulate_score,
    )

    # new pf without history
    pf_out2 = pfex.particle_accumulator(
        bm_model,
        subkey,
        y_meas,
        theta,
        n_particles,
        history=False,
        accumulator=accumulate_score,
    )

    # brute force calculation
    x_particles = pf_out1["x_particles"]
    ancestors = pf_out1["resample_out"]["ancestors"]
    logw = pf_out1["logw"][n_obs - 1]
    acc_out = pfex.accumulate_smooth(
        logw=logw,
        x_particles=x_particles,
        ancestors=ancestors,
        y_meas=y_meas,
        theta=theta,
        accumulator=accumulate_score,
    )
    max_diff = {"score_acc": rel_err(acc_out, pf_out2["accumulate_out"])}
    print_dict(max_diff)

if False:

    def accumulate_diff(x_prev, x_curr, y_curr, theta):
        r"""
        Accumulator for both score and hessian.
        """
        grad_meas = jax.grad(bm_model.meas_lpdf, argnums=2)
        grad_state = jax.grad(bm_model.state_lpdf, argnums=2)
        hess_meas = jax.jacfwd(jax.jacrev(bm_model.meas_lpdf, argnums=2), argnums=2)
        hess_state = jax.jacfwd(jax.jacrev(bm_model.state_lpdf, argnums=2), argnums=2)
        alpha = grad_meas(y_curr, x_curr, theta) + grad_state(x_curr, x_prev, theta)
        beta = hess_meas(y_curr, x_curr, theta) + hess_state(x_curr, x_prev, theta)
        return (alpha, beta)

    # new pf with history
    pf_out1 = pfex.particle_accumulator(
        bm_model,
        subkey,
        y_meas,
        theta,
        n_particles,
        history=True,
        accumulator=accumulate_diff,
    )

    # new pf without history
    pf_out2 = pfex.particle_accumulator(
        bm_model,
        subkey,
        y_meas,
        theta,
        n_particles,
        history=False,
        accumulator=accumulate_diff,
    )

    # brute force calculation
    x_particles = pf_out1["x_particles"]
    ancestors = pf_out1["resample_out"]["ancestors"]
    logw = pf_out1["logw"][n_obs - 1]
    acc_out = pfex.accumulate_smooth(
        logw=logw,
        x_particles=x_particles,
        ancestors=ancestors,
        y_meas=y_meas,
        theta=theta,
        accumulator=accumulate_diff,
    )
    max_diff = {
        "score_acc": rel_err(acc_out[0], pf_out2["accumulate_out"][0]),
        "hessian_acc": rel_err(acc_out[1], pf_out2["accumulate_out"][1]),
    }
    print_dict(max_diff)


if False:

    def accumulate_deriv(x_prev, x_curr, y_curr, theta):
        r"""
        Accumulator for both score and hessian.
        """
        grad_meas = jax.grad(bm_model.meas_lpdf, argnums=2)
        grad_state = jax.grad(bm_model.state_lpdf, argnums=2)
        hess_meas = jax.jacfwd(jax.jacrev(bm_model.meas_lpdf, argnums=2), argnums=2)
        hess_state = jax.jacfwd(jax.jacrev(bm_model.state_lpdf, argnums=2), argnums=2)
        alpha = grad_meas(y_curr, x_curr, theta) + grad_state(x_curr, x_prev, theta)
        beta = hess_meas(y_curr, x_curr, theta) + hess_state(x_curr, x_prev, theta)
        return (alpha, beta)

    job_descr = expand_grid(
        history=jnp.array([True]),  # False doesn't calculate the right thing
        score=jnp.array([False, True]),
        fisher=jnp.array([False, True]),
    )
    for i in jnp.arange(job_descr["history"].size):
        history = job_descr["history"][i]
        score = job_descr["score"][i]
        fisher = job_descr["fisher"][i]
        print("history = {}, score = {}, fisher = {}".format(history, score, fisher))
        # accumulator with history
        pf_out1 = pfex.particle_filter(
            bm_model,
            subkey,
            y_meas,
            theta,
            n_particles,
            score=False,
            fisher=False,
            history=True,
        )
        # auxillary filter with history
        pf_out2 = pfex.auxillary_filter_linear(
            bm_model,
            subkey,
            y_meas,
            theta,
            n_particles,
            score=score,
            fisher=fisher,
            history=history,
        )

        # check x_particles and logw
        max_diff = {k: rel_err(pf_out1[k], pf_out2[k]) for k in ["x_particles", "logw"]}
        print_dict(max_diff)

        # check ancestors
        max_diff = {
            k: rel_err(pf_out1["resample_out"][k], pf_out2[k]) for k in ["ancestors"]
        }
        print_dict(max_diff)

        # check loglik
        max_diff = {"loglik": rel_err(pf_out1["loglik"], pf_out2["loglik"])}

        if score or fisher:
            # score and hess using smoothing accumulator
            if history:
                x_particles = pf_out1["x_particles"]
                ancestors = pf_out1["resample_out"]["ancestors"]
                logw = pf_out1["logw"][n_obs - 1]
                alpha, beta = pfex.accumulate_smooth(
                    logw=logw,
                    x_particles=x_particles,
                    ancestors=ancestors,
                    y_meas=y_meas,
                    theta=theta,
                    accumulator=accumulate_diff,
                    mean=False,
                )
                prob = pf.logw_to_prob(logw)
                _score = jax.vmap(jnp.multiply)(prob, alpha)
                _hess = jax.vmap(lambda p, a, b: p * (jnp.outer(a, a) + b))(
                    prob, alpha, beta
                )
                _score, _hess = jtu.tree_map(
                    lambda x: jnp.sum(x, axis=0), (_score, _hess)
                )
            else:
                # score and hess using filtering accumulator
                _score, _hess = pf_out1["accumulate_out"]
            max_diff["score"] = rel_err(_score, pf_out2["score"])
            if fisher:
                max_diff["fisher"] = rel_err(
                    _hess - jnp.outer(_score, _score), pf_out2["fisher"]
                )

        print_dict(max_diff)

if False:
    # test linear vs quad
    # update: no longer relevant since underlying PF is now different
    job_descr = expand_grid(history=jnp.array([False, True]))
    for i in jnp.arange(job_descr["history"].size):
        history = job_descr["history"][i]
        print("history = {}".format(history))
        # auxillary filter linear
        pf_out1 = pfex.auxillary_filter_linear(
            bm_model,
            subkey,
            y_meas,
            theta,
            n_particles,
            score=False,
            fisher=False,
            history=history,
        )
        # auxillary filter quadratic
        pf_out2 = pfex.auxillary_filter_quad(
            bm_model,
            subkey,
            y_meas,
            theta,
            n_particles,
            score=False,
            fisher=False,
            history=history,
            tilde_for=False,
        )
        max_diff = {
            k: rel_err(pf_out1[k], pf_out2[k]) for k in ["loglik", "x_particles"]
        }
        print_dict(max_diff)
