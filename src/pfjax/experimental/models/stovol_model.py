import jax
import jax.numpy as jnp
import jax.scipy as jsp
from pfjax.experimental.sde_model import SDEModel


class StoVolModel(SDEModel):
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

    def __init__(self, dt, n_res, eps=0.1, unconstrained_scale=False):
        """
        Parameters
        ----------
        dt : float
            Time interval between measurements.
        n_res : int
            The resolution number, i.e., latent SDE variables are at intervals of `dt / n_res`.
        eps : float
            Standard deviation of noise in measurement model.
        unconstrained_scale : bool
            If `True`, the parameter vector is `theta = [alpha, log(gamma), eta, log(sigma), arctan(rho)]`.
        """
        super().__init__(dt, n_res, diff_diag=False, bootstrap=False)
        self._eps = eps
        self._unconstrained_scale = unconstrained_scale

    def to_unconstrained(self, constrained):
        """
        Convert parameters to unconstrained scale.
        """
        alpha, gamma, eta, sigma, rho = constrained
        return jnp.array([alpha, jnp.log(gamma), eta, jnp.log(sigma), jnp.arctanh(rho)])

    def to_constrained(self, unconstrained):
        """
        Convert parameters to constrained scale.
        """
        alpha, log_gamma, eta, log_sigma, atanh_rho = unconstrained
        return jnp.array(
            [alpha, jnp.exp(log_gamma), eta, jnp.exp(log_sigma), jnp.tanh(atanh_rho)]
        )

    def get_pars(self, pars):
        """
        Return the original parameters from whichever scale is being used.
        """
        theta = to_unconstrained(pars) if self._unconstrained_scale else pars
        return theta

    def drift(self, x, theta):
        alpha, gamma, eta, sigma, rho = self.get_pars(theta)
        return jnp.array([alpha - 0.5 * jnp.exp(2 * x[1]), -gamma * x[1] + eta])

    def diff(self, x, theta):
        alpha, gamma, eta, sigma, rho = self.get_pars(theta)
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
        return Y, A, Omega

    def meas_pars(self, x_curr, theta):
        _, A, Omega = self.bridge_pars(y_curr=None, theta=theta)
        mu = jnp.squeeze(jnp.dot(A, x_curr[-1]))
        sigma = jnp.squeeze(jnp.sqrt(Omega))
        return mu, sigma

    def meas_sample(self, key, x_curr, theta):
        # just return the last log-asset value
        # x_curr is an array of shape `(n_res, 2)`.
        mu, sigma = self.meas_pars(x_curr=x_curr, theta=theta)
        return mu + sigma * jax.random.normal(key)

    def meas_lpdf(self, y_curr, x_curr, theta):
        return 0.0
        mu, sigma = self.meas_pars(x_curr=x_curr, theta=theta)
        return jsp.stats.norm.logpdf(x=y_curr, loc=mu, scale=sigma)

    def pf_init(self, key, y_init, theta):
        """
        The implied prior distribution is
        ```
        z_init ~ stationary distribution
        x_init ~ flat prior
        ```
        The posterior distribution is thus
        ```
        z_init ~ stationary distribution
        x_init ~ Normal(y_init, eps^2)
        ```
        with the appropriate zero padding.
        """
        alpha, gamma, eta, sigma, rho = self.get_pars(theta)
        key_x, key_z = jax.random.split(key)

        ## First do sampling for Z_0. We shall sample Z0 according to the stationary distribution
        ## Z ~ N(eta / gamma, sigma^2 / (2 * gamma))
        z_sample = eta / gamma + sigma / jnp.sqrt(gamma**2) * jax.random.normal(
            key=key_z
        )

        ## Now draw X0 based on Y0
        ## In order for a consistent weight, we sample X0 ~ N(Y0/2, eps^2/2)
        x_sample = y_init + self._eps * jax.random.normal(key=key_x)

        # state_init = jnp.array([[X0_sample, Z0_sample]])
        state_init = self.prior_pad(jnp.array([x_sample, z_sample]))
        logw = 0.0
        return state_init, logw
