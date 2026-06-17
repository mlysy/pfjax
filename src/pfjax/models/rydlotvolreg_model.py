import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import lax
from pfjax import sde as sde


# --- main functions -----------------------------------------------------------

class RyderLotVolModel(sde.SDEModel):
    r"""
    Lotka-Volterra predator-prey model as in Ryder et al. (2018).

    The base model is:

    ::
        x_0 = (H_0, L_0)
        mu_t = (alpha H_t - beta H_t L_t, beta H_t L_t - gamma L_t)
        Sigma_t = [alpha H_t + beta H_t L_t, -beta H_t L_t ;
                   -beta H_t L_t, gamma L_t + beta H_t L_t]
        x_t = x_{t-1} + mu_t dt + Sigma_t^{1/2} sqrt(dt) z_t
    
    where z_t ~ N(0, 1). Note that both latent and model variables must be positive. Ryder VI model ensures x is always positive, and so they prefer using the SDE on the regular scale instead of log.

    - Model parameters: `theta = (alpha, beta, gamma, tau_L, H_0, L_0)`.
    - Global constants: `dt` and `n_res`, i.e., `m`.
    - State dimensions: `n_state = (n_res, 2)`.
    - Measurement dimensions: `n_meas = 1`.


    Args:
        dt: SDE interobservation time.
        n_res: SDE resolution number.  There are `n_res` latent variables per observation, equally spaced with interobservation time `dt/n_res`.
    """

    def __init__(self, dt, n_res):
        # creates "private" variables self._dt and self._n_res
        super().__init__(dt, n_res, diff_diag=False)
        self._n_state = (self._n_res, 2)
        self._n_meas = 2

    def drift(self, x, theta):
        r"""
        Calculates the SDE drift function on original scale.
        """
        alpha = theta[0]
        beta = theta[1]
        gamma = theta[2]
        return jnp.array([alpha * x[0] - beta * x[0] * x[1], 
                          beta * x[0] * x[1] - gamma * x[1]])
    
    def diff(self, x, theta):
        """
        Calculate the diffusion matrix on the original scale.
        """
        alpha = theta[0]
        beta = theta[1]
        gamma = theta[2]
        Sigma11 = alpha * x[0] + beta * x[0] * x[1]
        Sigma12 = - beta * x[0] * x[1]
        Sigma22 = gamma * x[1] +  beta * x[0] * x[1] 
        return jnp.array([[Sigma11, Sigma12],
                          [Sigma12, Sigma22]])


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
        # tau = theta[3:5]
        tau = jnp.ones(2)
        return jnp.sum(
            jsp.stats.norm.logpdf(y_curr,
                                  loc=x_curr[-1], scale=tau)
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
        # tau = theta[3:5]
        tau = jnp.ones(2)
        return x_curr[-1] + \
            tau * random.normal(key, (self._n_meas,))

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
        # x_init = jnp.log(theta[5:7])
        x_init = jnp.array([71., 79.])
        logw = 0.0
        return \
            jnp.append(jnp.zeros((self._n_res-1,) + x_init.shape),
                       jnp.expand_dims(x_init, axis=0), axis=0), \
            logw


    # def pf_step(self, key, x_prev, y_curr, theta):
    #     """
    #     Get particles at subsequent steps.

    #     Args:
    #         x_prev: State variable at previous time `t-1`.
    #         y_curr: Measurement variable at current time `t`.
    #         theta: Parameter value.
    #         key: PRNG key.

    #     Returns:

    #         Tuple:

    #         - x_curr: Sample of the state variable at current time `t`: `x_curr ~ q(x_curr)`.
    #         - logw: The log-weight of `x_curr`.
    #     """
    #     if self._bootstrap:
    #         x_curr, logw = super().pf_step(key, x_prev, y_curr, theta)
    #     else:
    #         tau = jnp.ones(2)
    #         omega = tau ** 2
    #         x_curr, logw = self.bridge_step(
    #             key, x_prev, y_curr, theta,
    #             y_curr, jnp.eye(2), jnp.diag(omega)
    #         )
    #     return x_curr, logw


    # def pf_init(self, key, y_init, theta):
    #     r"""
    #     Importance sampler for `x_init`.  

    #     See file comments for exact sampling distribution of `p(x_init | y_init, theta)`, i.e., we have a "perfect" importance sampler with `logw = CONST(theta)`.

    #     Args:
    #         key: PRNG key.
    #         y_init: Measurement variable at initial time `t = 0`.
    #         theta: Parameter value.

    #     Returns:

    #         Tuple:

    #         - x_init: A sample from the proposal distribution for `x_init`.
    #         - logw: The log-weight of `x_init`.
    #     """
    #     tau = jnp.ones(2)
    #     key, subkey = random.split(key)
    #     x_init = jnp.log(y_init + tau * random.truncated_normal(
    #         subkey,
    #         lower=-y_init/tau,
    #         upper=jnp.inf,
    #         shape=(self._n_state[1],)
    #     ))
    #     logw = jnp.sum(jsp.stats.norm.logcdf(y_init/tau))
    #     return \
    #         jnp.append(jnp.zeros((self._n_res-1,) + x_init.shape),
    #                    jnp.expand_dims(x_init, axis=0), axis=0), \
    #         logw
