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
