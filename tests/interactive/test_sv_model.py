import jax
import jax.numpy as jnp


class StoVolModel(sde.SDEModel):
    def __init__(self, dt):
        super().__init__(dt, n_res=2, diff_diag=False)

    def drift(self, x, theta):
        alpha = theta[0]
        gamma = theta[1]
        eta = theta[2]
        sigma = theta[3]
        rho = theta[4]
        return jnp.array([alpha - 0.5 * jnp.exp(2 * x[1]), -gamma * x[1] + eta])

    def diff(self, x, theta):
        alpha = theta[0]
        gamma = theta[1]
        eta = theta[2]
        sigma = theta[3]
        rho = theta[4]
        return jnp.array(
            [
                [jnp.exp(2 * x[1]), rho * jnp.exp(x[1]) * sigma],
                [rho * jnp.exp(x[1]) * sigma, sigma**2],
            ]
        )

    def meas_sample(self, key, x_curr, theta):
        # just return the last log-asset value
        # x_curr is an array of shape `(n_res, 2)`.
        return x_curr[-1, 0]

    def meas_lpdf(self, y_meas, x_curr, theta):
        delta = jnp.abs(y_meas - x_curr[-1, 0])
        # method 1: with error checking
        return jnp.where(delta > 1e-10, -jnp.inf, 0.0)
        # method 2: without error checking (hope for the best)
        return 0.0


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
