import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import lax
from pfjax import sde as sde


# --- main functions -----------------------------------------------------------

class StoVolModel(sde.SDEModel):
    r"""
    Stochastic Volatility model (exponential OU) as in Fang et al. (2020).

    The base model is:

        d X_t = (alpha - 1/2 e^{2Z_t}) dt + e^{Z_t} dB^X_t
        d Z_t = gamma(mu - Z_t) dt + sigma dB^Z_t
    
    where cor(dB^X_t, dB^Z_t) = rho. Here Z_t is log-volatility and X_t is log-price. Only X_t is observed
    and we assume the measurement model to be X_t with little noise.

    - Model parameters: `theta = (alpha, gamma, mu, sigma, rho)`.
    - Global constants: `dt` and `n_res`.
    - State dimensions: `n_state = (n_res, 2)`.
    - Measurement dimensions: `n_meas = 2`.

    Args:
        dt: SDE interobservation time.
        n_res: SDE resolution number.  There are `n_res` latent variables per observation, equally spaced with interobservation time `dt/n_res`.
    """

    def __init__(self, dt, n_res, bootstrap=True):
        # creates "private" variables self._dt and self._n_res
        super().__init__(dt, n_res, diff_diag=False)
        self._n_state = (self._n_res, 2)
        self._n_meas = 1
        self._bootstrap = bootstrap
        self._eps = 1e-3
        # self._eps = 2e-2


    def drift(self, x, theta):
        """
        Calculates the SDE drift function.
        """
        alpha, gamma, mu = theta[0:3]
        e2Z = jnp.exp(2*x[1])
        dx_drift = alpha - .5 * e2Z
        dz_drift = gamma * (mu - x[1])
        # dz_drift = mu - gamma * x[1]
        return jnp.array([dx_drift, dz_drift])

    def diff(self, x, theta):
        """
        Calculates the SDE diffusion function.
        """
        sigma, rho = theta[3:]
        eZ = jnp.exp(x[1])
        Sigma_XX = eZ * eZ
        Sigma_XZ = rho * sigma * eZ
        Sigma_ZZ = sigma * sigma
        return jnp.array([[Sigma_XX, Sigma_XZ],
                          [Sigma_XZ, Sigma_ZZ]])

    def meas_lpdf(self, y_curr, x_curr, theta):
        """
        Log-density of `p(y_curr | x_curr, theta)`.

        Args:
            y_curr: Measurement variable at current time `t`.
            x_curr: State variable at current time `t`.
            theta: Parameter value.

        Returns:
            The log-density of `p(y_curr | x_curr, theta)`.
        """
        # return jax.lax.cond(jnp.isclose(y_curr, x_curr[-1, 0]), lambda : 0.0, lambda : -jnp.inf)
        return jnp.sum(
            jsp.stats.norm.logpdf(y_curr,
                                  loc=x_curr[-1][0], scale=self._eps)
        )

    def meas_sample(self, key, x_curr, theta):
        """
        Sample from `p(y_curr | x_curr, theta)`.

        Args:
            x_curr: State variable at current time `t`.
            theta: Parameter value.
            key: PRNG key.

        Returns:
            Sample of the measurement variable at current time `t`: `y_curr ~ p(y_curr | x_curr, theta)`.
        """
        # return x_curr[-1, 0]
        return x_curr[-1][0] + self._eps * random.normal(key, (self._n_meas,))
     
    def pf_init(self, key, y_init, theta):
        r"""
        We use a delta function for `x_init`, ie, `p(x_init = theta[5:7]) = 1`.

        Args:
            key: PRNG key.
            y_init: Measurement variable at initial time `t = 0`.
            theta: Parameter value.

        Returns:

            Tuple:

            - x_init: A sample from the proposal distribution for `x_init`.
            - logw: The log-weight of `x_init`.
        """
        # key, subkey = random.split(key)
        x_init = jnp.array([jnp.log(1000), 0.1])
        logw = 0.0
        return \
            jnp.append(jnp.zeros((self._n_res-1,) + x_init.shape),
                       jnp.expand_dims(x_init, axis=0), axis=0), \
            logw
    
    def pf_step(self, key, x_prev, y_curr, theta):
        """
        Get particles at subsequent steps.

        Args:
            x_prev: State variable at previous time `t-1`.
            y_curr: Measurement variable at current time `t`.
            theta: Parameter value.
            key: PRNG key.

        Returns:

            Tuple:

            - x_curr: Sample of the state variable at current time `t`: `x_curr ~ q(x_curr)`.
            - logw: The log-weight of `x_curr`.
        """
        if self._bootstrap:
            x_curr, logw = super().pf_step(key, x_prev, y_curr, theta)
        else:
            x_curr, logw = self.bridge_step(
                key, x_prev, y_curr, theta,
                y_curr, jnp.array([[1., 0.]]), self._eps**2 * jnp.ones((1,1))
            )
        return x_curr, logw
        
