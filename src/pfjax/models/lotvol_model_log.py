"""
Lotka-Volterra predator-prey model on the log-scale, using Ito's lemma.

The model on the regular scale is:

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

After applying Ito's lemma, the model on the log-scale has drift and diffusion matrices:
Uses It's lemma formulation from examples/sde.ipynb
drift: 

"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import lax
from pfjax import sde as sde

# --- helper functions ---------------------------------------------------------


def lotvol_drift(x, dt, theta):
    """
    Calculates the SDE drift function.
    """
    alpha = theta[0]
    beta = theta[1]
    gamma = theta[2]
    delta = theta[3]
    return x + jnp.array([alpha - beta * jnp.exp(x[1]),
                          -gamma + delta * jnp.exp(x[0])]) * dt


# --- main functions -----------------------------------------------------------

class LotVolModelLog(sde.SDEModel):
    def __init__(self, dt, n_res):
        super().__init__(dt, n_res, diff_diag=True)
        self._n_state = (self._n_res, 2)

    def drift(self, x, theta):
        """
        Calculates the SDE drift on the regular scale
        Args:
            x: 
            theta: parameter values on the log-scale
        """
        alpha = jnp.exp(theta[0])
        beta = jnp.exp(theta[1])
        gamma = jnp.exp(theta[2])
        delta = jnp.exp(theta[3])
        return jnp.array([alpha - beta * jnp.exp(x[1]),
                          -gamma + delta * jnp.exp(x[0])])

    def diff(self, x, theta):
        """
        Calculates the SDE diffusion function on the regular scale 
        Args:
            x: 
            theta: parameter values on the log-scale
        """
        return jnp.exp(theta[4:6])

    # def drift (self, x, theta):
    #     """ 
    #     Calculates the SDE drift on the log-scale
    #     Uses It's lemma formulation from examples/sde.ipynb
    #     """
    #     A = 1/x
    #     reg_diff = self._diff(x, theta) # diffusion on the regular scale
    #     reg_drift = self._drift(x, theta) # drift on regular scale
    #     b = -0.5 * (1/x**2) * (reg_diff**2)
    #     return (A * reg_drift) + b

    # def diff (self, x, theta):
    #     """ 
    #     Calculates the SDE diffusion matrix on the log-scale
    #     """
    #     A = 1/x #jnp.diag([1/x[0], 1/x[1]])
    #     return A * self._diff(x, theta)

      
    def state_lpdf_for(self, x_curr, x_prev, theta):
        """
        Calculates the log-density of `p(x_curr | x_prev, theta)`.
        For-loop version for testing.
        Args:
            x_curr: State variable at current time `t`.
            x_prev: State variable at previous time `t-1`.
            theta: Parameter value.
        Returns:
            The log-density of `p(x_curr | x_prev, theta)`.
        """
        dt_res = self._dt/self._n_res
        x0 = jnp.append(jnp.expand_dims(
            x_prev[self._n_res-1], axis=0), x_curr[:self._n_res-1], axis=0)
        x1 = x_curr
        sigma = theta[4:6] * jnp.sqrt(dt_res)
        lp = jnp.array(0.0)
        for t in range(self._n_res):
            lp = lp + jnp.sum(jsp.stats.norm.logpdf(
                x1[t],
                loc=lotvol_drift(x0[t], dt_res, theta),
                scale=sigma
            ))
        return lp

    def state_sample_for(self, key, x_prev, theta):
        """
        Samples from `x_curr ~ p(x_curr | x_prev, theta)`.
        For-loop version for testing.
        Args:
            key: PRNG key.
            x_prev: State variable at previous time `t-1`.
            theta: Parameter value.
        Returns:
            Sample of the state variable at current time `t`: `x_curr ~ p(x_curr | x_prev, theta)`.
        """
        dt_res = self._dt/self._n_res
        sigma = theta[4:6] * jnp.sqrt(dt_res)
        x_curr = jnp.zeros(self._n_state)
        x_state = x_prev[self._n_res-1]
        for t in range(self._n_res):
            key, subkey = random.split(key)
            x_state = lotvol_drift(x_state, dt_res, theta) + \
                random.normal(subkey, (self._n_state[1],)) * sigma
            x_curr = x_curr.at[t].set(x_state)
        return x_curr

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
        tau = jnp.exp(theta[6:8])
        return jnp.sum(
            jsp.stats.norm.logpdf(y_curr,
                                  loc=jnp.exp(x_curr[-1]), scale=tau)
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
        tau = jnp.exp(theta[6:8])
        return jnp.exp(x_curr[-1]) + \
            tau * random.normal(key, (self._n_state[1],))

    def pf_init(self, key, y_init, theta):
        """
        Importance sampler for `x_init`.  
        See file comments for exact sampling distribution of `p(x_init | y_init, theta)`, i.e., we have a "perfect" importance sampler with `logw = CONST(theta)`.
        Args:
            key: PRNG key.
            y_init: Measurement variable at initial time `t = 0`.
            theta: Parameter value.
        Returns:
            - x_init: A sample from the proposal distribution for `x_init`.
            - logw: The log-weight of `x_init`.
        """
        tau = jnp.exp(theta[6:8])
        key, subkey = random.split(key)
        x_init = jnp.log(y_init + tau * random.truncated_normal(
            subkey,
            lower=-y_init/tau,
            upper=jnp.inf,
            shape=(self._n_state[1],)
        ))
        logw = jnp.sum(jsp.stats.norm.logcdf(y_init/tau))
        return \
            jnp.append(jnp.zeros((self._n_res-1,) + x_init.shape),
                       jnp.expand_dims(x_init, axis=0), axis=0), \
            logw
