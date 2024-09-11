import jax
import jax.numpy as jnp
import jax.random as random
import jax.tree_util as jtu
import jax.scipy
import pfjax as pf
import pfjax.models
from pfjax.experimental.continuous_time import ContinuousTimeModel


def tree_rel_err(X1, X2):
    """
    Relative error between two PyTrees.

    For each leaf, adds 0.1 to the denominator to avoid nan's when its equal to zero.
    """
    def rel_err(x1, x2):
        x1 = x1.ravel() * 1.0
        x2 = x2.ravel() * 1.0
        return jnp.max(jnp.abs(x1 - x2) / (0.1 + jnp.abs(x1)))
    return jtu.tree_map(rel_err, X1, (X2))


class LotVolModel(ContinuousTimeModel):

    def __init__(self, dt, n_res):
        super().__init__(dt=dt, n_res=n_res, meas_linear=False, bootstrap=True)
        self._n_state = (self._n_res, 2)

    def drift(self, x, theta):
        r"""
        Calculates the SDE drift function.
        """
        alpha = theta[0]
        beta = theta[1]
        gamma = theta[2]
        delta = theta[3]
        return jnp.array([alpha - beta * jnp.exp(x[1]),
                          -gamma + delta * jnp.exp(x[0])])

    def diff(self, x, theta):
        r"""
        Calculates the SDE diffusion function.
        """
        return theta[4:6]

    def state_dt_lpdf(self, x_curr, x_prev, dt, theta):
        dr = x_prev + self.drift(x_prev, theta) * dt
        df = self.diff(x_prev, theta) * jnp.sqrt(dt)
        return jax.scipy.stats.norm.logpdf(
            x=x_curr,
            loc=dr,
            scale=df
        )

    def state_dt_sim(self, key, x_prev, dt, theta):
        dr = x_prev + self.drift(x_prev, theta) * dt
        df = self.diff(x_prev, theta) * jnp.sqrt(dt)
        return dr + df * random.normal(key, (x_prev.shape[0],))

    def meas_lpdf(self, y_curr, x_curr, theta):
        r"""
        Log-density of `p(y_curr | x_curr, theta)`.

        Args:
            y_curr: Measurement variable at current time `t`.
            x_curr: State variable at current time `t`.
            theta: Parameter value.

        Returns:
            The log-density of `p(y_curr | x_curr, theta)`.
        """
        tau = theta[6:8]
        return jnp.sum(
            jax.scipy.stats.norm.logpdf(
                x=y_curr,
                loc=jnp.exp(x_curr[-1]),
                scale=tau
            )
        )

    def meas_sample(self, key, x_curr, theta):
        r"""
        Sample from `p(y_curr | x_curr, theta)`.

        Args:
            key: PRNG key.
            x_curr: State variable at current time `t`.
            theta: Parameter value.

        Returns:
            Sample of the measurement variable at current time `t`: `y_curr ~ p(y_curr | x_curr, theta)`.
        """
        tau = theta[6:8]
        return jnp.exp(x_curr[-1]) + \
            tau * random.normal(key, (self._n_state[1],))

    def pf_init(self, key, y_init, theta):
        r"""
        Importance sampler for `x_init`.  

        See file comments for exact sampling distribution of `p(x_init | y_init, theta)`, i.e., we have a "perfect" importance sampler with `logw = CONST(theta)`.

        Args:
            key: PRNG key.
            y_init: Measurement variable at initial time `t = 0`.
            theta: Parameter value.

        Returns:

            Tuple:

            - x_init: A sample from the proposal distribution for `x_init`.
            - logw: The log-weight of `x_init`.
        """
        tau = theta[6:8]
        key, subkey = random.split(key)
        x_init = jnp.log(y_init + tau * random.truncated_normal(
            subkey,
            lower=-y_init/tau,
            upper=jnp.inf,
            shape=(self._n_state[1],)
        ))
        logw = jnp.sum(jax.scipy.stats.norm.logcdf(y_init/tau))
        return \
            jnp.append(jnp.zeros((self._n_res-1,) + x_init.shape),
                       jnp.expand_dims(x_init, axis=0), axis=0), \
            logw


key = random.PRNGKey(0)
# parameter values
alpha = 1.02
beta = 1.02
gamma = 4.
delta = 1.04
sigma_H = .1
sigma_L = .2
tau_H = .25
tau_L = .35
theta = jnp.array([alpha, beta, gamma, delta, sigma_H, sigma_L, tau_H, tau_L])
# data specification
dt = .09
n_res = 10
n_obs = 7
x_init = jnp.block([[jnp.zeros((n_res-1, 2))],
                    [jnp.log(jnp.array([5., 3.]))]])

# class inherited from SDEModel
lv_model = pf.models.LotVolModel(dt=dt, n_res=n_res)
# class inherited from ContinuousTimeModel
lv_model2 = LotVolModel(dt=dt, n_res=n_res)

# test simulation
pf_simulate = jax.jit(
    fun=pf.simulate,
    static_argnames=("model", "n_obs")
)

out1 = pf_simulate(
    model=lv_model,
    key=key,
    n_obs=n_obs,
    x_init=x_init,
    theta=theta
)

out2 = pf_simulate(
    model=lv_model2,
    key=key,
    n_obs=n_obs,
    x_init=x_init,
    theta=theta
)

tree_rel_err(out1, out2)

y_meas, x_state = out1

# test loglik_full
pf_loglik_full = jax.jit(
    fun=pf.loglik_full,
    static_argnames="model"
)

out1 = pf_loglik_full(
    model=lv_model,
    y_meas=y_meas,
    x_state=x_state,
    theta=theta
)

out2 = pf_loglik_full(
    model=lv_model2,
    y_meas=y_meas,
    x_state=x_state,
    theta=theta
)

tree_rel_err(out1, out2)

# test bootstrap pf
n_particles = 5
pf_particle_filter = jax.jit(
    fun=pf.particle_filter,
    static_argnames=("model", "n_particles", "score", "fisher", "history")
)

out1 = pf_particle_filter(
    model=lv_model,
    key=key,
    y_meas=y_meas,
    theta=theta,
    n_particles=n_particles,
    score=True,
    fisher=True,
    history=True
)

out2 = pf_particle_filter(
    model=lv_model2,
    key=key,
    y_meas=y_meas,
    theta=theta,
    n_particles=n_particles,
    score=True,
    fisher=True,
    history=True
)

tree_rel_err(out1, out2)

# # bridge proposal
# lv_model.bridge_step(
#     key=key,
#     x_prev=x_state[0],
#     Y=jnp.log(y_meas[1]),
#     theta=theta,
#     A=jnp.eye(2),
#     Omega=jnp.eye(2)
# )


# simulate with inherited class
lv_model1 = pf.LotVolModel(dt=dt, n_res=n_res)
y_meas1, x_state1 = pf.simulate(lv_model1, key, n_obs, x_init, theta)
# simulate with non-inherited class
lv_model2 = lv.LotVolModel(dt=dt, n_res=n_res)
y_meas2, x_state2 = pf.simulate(lv_model2, key, n_obs, x_init, theta)

y_meas1 - y_meas2
x_state1 - x_state2

n_state = (n_res, 2)

key = random.PRNGKey(0)
lv_model = pf.LotVolModel(dt=dt, n_res=n_res)

key, subkey = random.split(key)
x_prev = random.normal(subkey, n_state)

key, subkey = random.split(key)
x_curr = lv_model.state_sample(x_prev, theta, key)
x_curr_for = lv_model.state_sample_for(x_prev, theta, key)

print("x_curr - x_curr_for = \n", x_curr - x_curr_for)

state_lp = lv_model.state_lpdf(x_curr, x_prev, theta)
state_lp_for = lv_model.state_lpdf_for(x_curr, x_prev, theta)
print("state_lp - state_lp_for= \n", state_lp - state_lp_for)

# --- particle filter ----------------------------------------------------------

pf.particle_filter(lv_model, key,
                   y_meas, theta, n_particles=5,
                   particle_sampler=pf.particle_resample_ot)

key, subkey = random.split(key)
x_init = init_sample(y_init=jnp.log(jnp.array([5., 3.])),
                     theta=jnp.append(theta[0:6], jnp.array([0., 0.])),
                     key=subkey)

n_obs = 100
key, subkey = random.split(key)
y_meas, x_state = meas_sim(n_obs, x_init, theta, subkey)
t_seq = jnp.arange(n_obs) * dt

plt.plot(t_seq, y_meas[:, 0])
# plt.show()

n_particles = 100
pf_out = particle_filter(y_meas, theta, n_particles, key)
