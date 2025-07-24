# jax
import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy as jsp

# pfjax
import pfjax as pf
from pfjax import sde as sde
from pfjax.experimental import sde_model as sde_test


# We will be giving two versions of sde model, one from standard definition and one from experiement
# Using the experiment sde moment will enable the use of particle_filter_rb
class StoVolModel(sde.SDEModel):
    """
    In a a stochastic volatility model, we have

        Latent, or state variables:
            X_t: The log-price of the derivative at time t
            Z_t: The volatility at time t
        Observed, or measurement vairables:
            Y_t: The noised log-price, i.e. Y_t = X_t + N()


    The interest of a stochastic volatility model lies in simulating both observed and latent variables
    and from there do inferences on parameters in the model

    """

    def __init__(self, n_res, dt, eps=0.1, bootstrap=False):
        super().__init__(dt, n_res, diff_diag=False)
        self._eps = eps

    def prior_pad(self, x_init):
        """
        Pad the initial state with `n_res - 1` zeros.

        This is helpful for constructing `x_init` to have the correct dimension.
        """

        zeros = jnp.zeros((self._n_res - 1, 2))
        return jnp.concatenate([zeros, x_init[None]])

    def drift(self, x, theta):
        alpha, gamma, eta, sigma, rho = theta
        return jnp.array([alpha - 0.5 * jnp.exp(2 * x[1]), -gamma * x[1] + eta])

    def diff(self, x, theta):
        alpha, gamma, eta, sigma, rho = theta
        return jnp.array(
            [
                [jnp.exp(2 * x[1]), rho * jnp.exp(x[1]) * sigma],
                [rho * jnp.exp(x[1]) * sigma, sigma**2],
            ]
        )

    def bridge_pars(self, y_curr, theta):
        A = jnp.atleast_2d(jnp.array([1.0, 0.0]))
        Omega = jnp.atleast_2d(self._eps**2)
        Y = y_curr
        return A, Omega, Y

    def meas_pars(self, x_curr, theta):
        A, Omega, _ = self.bridge_pars(y_curr=None, theta=theta)
        mu = jnp.squeeze(jnp.dot(A, x_curr[-1]))
        sigma = jnp.squeeze(jnp.sqrt(Omega))
        return mu, sigma

    def meas_sample(self, key, x_curr, theta):
        # just return the last log-asset value
        # x_curr is an array of shape `(n_res, 2)`.
        mu, sigma = self.meas_pars(x_curr=x_curr, theta=theta)
        return mu + sigma * jax.random.normal(key)

    def meas_lpdf(self, y_curr, x_curr, theta):
        mu, sigma = self.meas_pars(x_curr=x_curr, theta=theta)
        return jsp.stats.norm.logpdf(x=y_curr, loc=mu, scale=sigma)

    def pf_step(self, key, x_prev, y_curr, theta):
        A, Omega, Y = self.bridge_pars(y_curr=y_curr, theta=theta)
        return self.bridge_step(
            key=key, x_prev=x_prev, y_curr=y_curr, theta=theta, Y=Y, A=A, Omega=Omega
        )

    def pf_init(self, key, y_init, theta):
        alpha, gamma, eta, sigma, rho = theta
        key_x, key_z = jax.random.split(key)

        ## First do sampling for Z_0. We shall sample Z0 according to the stationary distribution
        ## Z ~ N(eta / gamma, sigma^2 / (2 * gamma))
        Z0_sample = eta / gamma + sigma / jnp.sqrt(gamma**2) * jax.random.normal(
            key=key_z
        )

        ## Now draw X0 based on Y0
        ## In order for a consistent weight, we sample X0 ~ N(Y0/2, eps^2/2)
        X0_sample = y_init / 2 + self._eps / jnp.sqrt(2) * jax.random.normal(key=key_x)

        state_init = jnp.array([[X0_sample, Z0_sample]])
        logw = 0
        return state_init, logw


class StoVolModel_test(sde_test.SDEModel):
    ## This is a duplicate of the previous version but with step_sample function activiated so that we can use pf_rb
    def __init__(self, n_res, dt, eps=0.1):
        super().__init__(dt, n_res, diff_diag=False, bootstrap=False)
        self._eps = eps

    def drift(self, x, theta):
        alpha, gamma, eta, sigma, rho = theta
        return jnp.array([alpha - 0.5 * jnp.exp(2 * x[1]), -gamma * x[1] + eta])

    def diff(self, x, theta):
        alpha, gamma, eta, sigma, rho = theta
        return jnp.array(
            [
                [jnp.exp(2 * x[1]), rho * jnp.exp(x[1]) * sigma],
                [rho * jnp.exp(x[1]) * sigma, sigma**2],
            ]
        )

    ## In bridge_pars, changed the order from (A, Omega, Y) to (Y, A, Omega)
    ## so that it can fit _step_to_pars in sde_test
    def bridge_pars(self, y_curr, theta):
        A = jnp.atleast_2d(jnp.array([1.0, 0.0]))
        Omega = jnp.atleast_2d(self._eps**2)
        Y = y_curr
        return Y, A, Omega

    def meas_pars(self, x_curr, theta):
        Y, A, Omega = self.bridge_pars(y_curr=None, theta=theta)
        mu = jnp.squeeze(jnp.dot(A, x_curr[-1]))
        sigma = jnp.squeeze(jnp.sqrt(Omega))
        return mu, sigma

    def meas_sample(self, key, x_curr, theta):
        # just return the last log-asset value
        # x_curr is an array of shape `(n_res, 2)`.
        mu, sigma = self.meas_pars(x_curr=x_curr, theta=theta)
        return mu + sigma * jax.random.normal(key)

    def meas_lpdf(self, y_curr, x_curr, theta):
        mu, sigma = self.meas_pars(x_curr=x_curr, theta=theta)
        return jsp.stats.norm.logpdf(x=y_curr, loc=mu, scale=sigma)

    def pf_step(self, key, x_prev, y_curr, theta):
        Y, A, Omega = self.bridge_pars(y_curr=y_curr, theta=theta)
        return self.bridge_step(
            key=key, x_prev=x_prev, y_curr=y_curr, theta=theta, Y=Y, A=A, Omega=Omega
        )

    def pf_init(self, key, y_init, theta):
        alpha, gamma, eta, sigma, rho = theta
        key_x, key_z = jax.random.split(key)

        ## First do sampling for Z_0. We shall sample Z0 according to the stationary distribution
        ## Z ~ N(eta / gamma, sigma^2 / (2 * gamma))
        Z0_sample = eta / gamma + sigma / jnp.sqrt(gamma**2) * jax.random.normal(
            key=key_z
        )

        ## Now draw X0 based on Y0
        ## In order for a consistent weight, we sample X0 ~ N(Y0/2, eps^2/2)
        X0_sample = y_init / 2 + self._eps / jnp.sqrt(2) * jax.random.normal(key=key_x)

        state_init = jnp.array([[X0_sample, Z0_sample]])
        logw = 0
        return state_init, logw


# --- simulate data ------------------------------------------------------------

# true parameters
alpha = 0.001
gamma = 3
eta = 3 * -1.3
sigma = 1
rho = -0.8
true_theta = jnp.array([alpha, gamma, eta, sigma, rho])

# data specification
n_res = 3
dt = 0.53
n_obs = 100
x_init = jnp.array([0.0, 0.0])
key = jax.random.PRNGKey(0)


# simulation
sv_model = StoVolModel(dt=dt, n_res=n_res)
sv_model_test = StoVolModel_test(dt=dt, n_res=n_res)
key, subkey = jax.random.split(key)
meas, state = pf.simulate(
    model=sv_model,
    key=subkey,
    n_obs=n_obs,
    x_init=sv_model.prior_pad(x_init),
    theta=true_theta,
)

meas_test, state_test = pf.simulate(
    model=sv_model_test,
    key=subkey,
    n_obs=n_obs,
    x_init=sv_model_test.prior_pad(x_init),
    theta=true_theta,
)

print(jnp.min(jnp.abs(meas - meas_test)))
print(jnp.min(jnp.abs(state - state_test)))

# --- check that bridge proposal is the same for sv_model and sv_model_test ----

n = 2
A, Omega, Y = sv_model.bridge_pars(
    y_curr=meas[0],
    theta=true_theta,
)
A, Omega, Y
dt_prev = dt / n_res
dt_next = dt - (n + 1) * dt / n_res
mv_pars = sv_model._bridge_mv(
    x=state[0][0],
    n=n,
    Y=Y,
    A=A,
    Omega=Omega,
    theta=true_theta,
)
mv_pars_test = sv_model_test._step_dt_pars(
    x_prev=state[0][0],
    y_next=meas[0],
    dt_prev=dt_prev,
    dt_next=dt_next,
    theta=true_theta,
)
print(jnp.min(jnp.abs(mv_pars[0] - mv_pars_test[0])))
print(jnp.min(jnp.abs(mv_pars[1] - mv_pars_test[1])))


# --- old ----------------------------------------------------------------------

import jax
import jax.numpy as jnp


class StoVolModel(sde.SDEModel):
    def __init__(self, n_res, dt, x_init, eps=0.00001):
        """
        Parameters
        ----------
        x_init: jax.Array
            Vector of length two specifying the initial value of the SDE (considered fixed and known).
        """
        super().__init__(dt, n_res, diff_diag=False, bootstrap=False)
        self._eps = eps

    def drift(self, x, theta):
        alpha, gamma, eta, sigma, rho = theta
        return jnp.array([alpha - 0.5 * jnp.exp(2 * x[1]), -gamma * x[1] + eta])

    def diff(self, x, theta):
        alpha, gamma, eta, sigma, rho = theta
        return jnp.array(
            [
                [jnp.exp(2 * x[1]), rho * jnp.exp(x[1]) * sigma],
                [rho * jnp.exp(x[1]) * sigma, sigma**2],
            ]
        )

    def bridge_pars(self, y_curr, theta):
        A = jnp.array([1.0, 0.0])
        Omega = eps**2
        Y = y_curr
        return A, Omega, Y

    def meas_pars(self, x_curr, theta):
        A, Omega, _ = self.bridge_pars(y_curr=None, theta=theta)
        mu = jnp.dot(A, x_curr[-1])
        sigma = jnp.sqrt(Omega)
        return mu, sigma

    def meas_sample(self, key, x_curr, theta):
        # just return the last log-asset value
        # x_curr is an array of shape `(n_res, 2)`.
        mu, sigma = self.meas_pars(x_curr=x_curr, theta=theta)
        return mu + sigma * jax.random.normal(key)

    def meas_lpdf(self, x_curr, y_curr, theta):
        mu, sigma = self.meas_pars(x_curr=x_curr, theta=theta)
        return jsp.stats.norm.logpdf(x=y_curr, loc=mu, scale=sigma)

    def pf_step(self, key, x_prev, y_curr, theta):
        A, Omega, Y = self.bridge_pars(y_curr=y_curr, theta=theta)
        return self.bridge_step(
            key=key, x_prev=x_prev, y_curr=y_curr, theta=theta, Y=Y, A=A, Omega=Omega
        )


# true parameters
alpha = 0.03
gamma = 0.01
eta = 0.02
sigma = 0.04
rho = 0.3
theta_true = jnp.array([alpha, gamma, eta, sigma, rho])

# data specification
dt = 0.5
n_obs = 100
x_init = jnp.array([0.0, 0.0])
key = jax.random.PRNGKey(0)

# simulation
sv_model = StoVolModel(dt=dt)
sv_model.state_sample(key=key, x_prev=x_init, theta=theta_true)

a = pfjax.sde.SDEModel(dt=0.1, n_res=1, diff_diag=True)
b = pfjax.sde.SDEModel(dt=0.1, n_res=1, diff_diag=False)
