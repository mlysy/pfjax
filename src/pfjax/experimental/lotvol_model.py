r"""
Lotka-Volterra predator-prey model.

The model is:

```
exp(x_m0) = exp( (logH_m0, logL_m0) ) ~ pi(x_m0) \propto 1
logH_mt ~ N(logH_{m,t-1} + (alpha - beta exp(logL_{m,t-1})) dt/m,
            sigma_H^2 dt/m)
logL_mt ~ N(logL_{m,t-1} + (-gamma + delta exp(logH_{m,t-1})) dt/m,
            sigma_L^2 dt/m)
y_t ~ N( exp(x_{m,mt}), diag(tau_H^2, tau_L^2) )
```

- Model parameters: `theta = (alpha, beta, gamma, delta, sigma_H, sigma_L, tau_H, tau_L)`.
- Global constants: `dt` and `n_res`, i.e., `m`.
- State dimensions: `n_state = (n_res, 2)`.
- Measurement dimensions: `n_meas = 2`.

**Notes:**

- The measurement `y_t` corresponds to `x_t = (x_{m,(t-1)m+1}, ..., x_{m,tm})`, i.e., aligns with the last element of `x_t`.

- The prior is such that `p(x_0 | y_0, theta)` is given by:

    ```
    exp(x_{m0}) ~ TruncatedNormal( y_0, diag(tau_H^2, tau_L^2) )
    x_{m,n} = 0 for n = -m+1, ..., -1.
    ```

"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from pfjax.experimental.sde_model import SDEModel


class LotVolModel(SDEModel):
    def __init__(self, dt, n_res):
        super().__init__(
            dt=dt,
            n_res=n_res,
            diff_diag=True,
            bootstrap=True,
        )

    def drift(self, x, theta):
        """
        Calculates the SDE drift function.
        """
        alpha = theta[0]
        beta = theta[1]
        gamma = theta[2]
        delta = theta[3]
        return jnp.array([alpha - beta * jnp.exp(x[1]), -gamma + delta * jnp.exp(x[0])])

    def diff(self, x, theta):
        """
        Calculates the SDE diffusion function.
        """
        tau = theta[4:6]
        return tau

    def meas_lpdf(self, y_curr, x_curr, theta):
        """
        Log-density of `p(y_curr | x_curr, theta)`.

        Args:
            y_curr: Measurement variable at current time `t`.
            x_curr: State variable at current time `t`.
            theta: Parameter value.

        Returns
            The log-density of `p(y_curr | x_curr, theta)`.
        """
        tau = theta[6:8]
        return jnp.sum(
            jsp.stats.norm.logpdf(y_curr, loc=jnp.exp(x_curr[-1]), scale=tau)
        )

    def meas_sample(self, key, x_curr, theta):
        """
        Sample from `p(y_curr | x_curr, theta)`.

        Args:
            key: PRNG key.
            x_curr: State variable at current time `t`.
            theta: Parameter value.

        Returns:
            Sample of the measurement variable at current time `t`: `y_curr ~ p(y_curr | x_curr, theta)`.
        """
        tau = theta[6:8]
        return jnp.exp(x_curr[-1]) + tau * jax.random.normal(key=key, shape=(2,))

    def pf_init(self, key, y_init, theta):
        """
        Particle filter calculation for `x_init`.

        Samples from an importance sampling proposal distribution
        ```
        x_init ~ q(x_init) = q(x_init | y_init, theta)
        ```
        and calculates the log weight
        ```
        logw = log p(y_init | x_init, theta) + log p(x_init | theta) - log q(x_init)
        ```

        **FIXME:** Explain what the proposal is and why it gives `logw = 0`.

        In fact, if you think about it hard enough then it's not actually a perfect proposal...

        Args:
            y_init: Measurement variable at initial time `t = 0`.
            theta: Parameter value.
            key: PRNG key.

        Returns:
            - x_init: A sample from the proposal distribution for `x_init`.
            - logw: The log-weight of `x_init`.
        """
        tau = theta[6:8]
        key, subkey = jax.random.split(key)
        x_init = jax.random.truncated_normal(
            key=subkey, lower=-y_init / tau, upper=jnp.inf, shape=(2,)
        )
        x_init = jnp.log(y_init + tau * x_init)
        logw = jnp.sum(jsp.stats.norm.logcdf(y_init / tau))
        return (
            jnp.append(
                jnp.zeros((self._n_res - 1,) + x_init.shape),
                jnp.expand_dims(x_init, axis=0),
                axis=0,
            ),
            logw,
        )
