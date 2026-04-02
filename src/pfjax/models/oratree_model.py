import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import lax
from pfjax import sde as sde


# --- main functions -----------------------------------------------------------

class OraTreeModel(sde.SDEModel):
    r"""
    The orange tree growth model (Picchini et al., 2020) is given by

        d X_t^i = 1/(phi_1^i phi_2^i) X_t^i (phi_1^i - X_t^i) dt + sigma sqrt(X_t^i) d B_t^i
    
    for i=1,2, ... N and phi_1^i ~ N(phi_1, sigma_1), and phi_2^i ~ N(phi_2, sigma_2)
    and we assume the measurement model to be X_t to have unit variance.

    - Model parameters: `theta = (phi_1, phi_2, sigma_1, sigma_2, sigma)`.
    - Global constants: `dt` and `n_res`.
    - State dimensions: `n_state = (n_res, N)`.
    - Measurement dimensions: `n_meas = N`.

    Args:
        dt: SDE interobservation time.
        n_res: SDE resolution number.  There are `n_res` latent variables per observation, equally spaced with interobservation time `dt/n_res`.
    """

    def __init__(self, dt, n_res, N, bootstrap=True):
        # creates "private" variables self._dt and self._n_res
        super().__init__(dt, n_res, diff_diag=False)
        self._N = N
        self._n_state = (self._n_res, N)
        self._n_meas = N
        self._bootstrap = bootstrap
        self._eps = 1.

    def drift(self, x, theta):
        r"""
        Calculates the SDE drift function on original scale.
        """
        phi1 = theta[:2*self._N][::2]
        phi2 = theta[:2*self._N][1::2]
        return 1/(phi1 * phi2) * x * (phi1 - x)

    def diff(self, x, theta):
        """
        Calculate the diffusion matrix on the original scale.
        """
        sigma = theta[-1]
        # return sigma * jnp.sqrt(x)
        return jnp.diag(x) * sigma * sigma
    
    # def drift(self, x, theta):
    #     """
    #     Calculates the SDE drift function on the log scale.
    #     """
    #     x = jnp.exp(x)
    #     mu = self._drift(x, theta)
    #     # Sigma_half = self._diff(x, theta)
    #     Sigma = self._diff(x, theta)

    #     f_p = 1/x
    #     f_pp = -1/x/x

    #     # mu_trans = f_p * mu + 0.5 * f_pp * Sigma_half * Sigma_half
    #     mu_trans = f_p * mu + 0.5 * f_pp * jnp.diag(Sigma)
    #     return mu_trans

    # def diff(self, x, theta):
    #     """
    #     Calculates the SDE diffusion function on the log scale.
    #     """
    #     x = jnp.exp(x)
    #     # Sigma_half = self._diff(x, theta)
    #     Sigma = self._diff(x, theta)

    #     f_p = 1/x
    #     Sigma_trans = jnp.outer(f_p, f_p) * Sigma
    #     # Sigma_half_trans = f_p * Sigma_half 
    #     return Sigma_trans
    
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
                                  loc=x_curr[-1], scale=self._eps)
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
        # return x_curr[-1]
        return x_curr[-1] + self._eps * random.normal(key, (self._n_meas,))
     
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
        x_init = jnp.array([30.] * self._N)
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
            tau = jnp.ones(self._N) * self._eps
            # omega = (tau / y_curr)**2
            omega = tau ** 2
            x_curr, logw = self.bridge_step(
                key, x_prev, y_curr, theta,
                y_curr, jnp.eye(self._N), jnp.diag(omega)
            )
        return x_curr, logw

