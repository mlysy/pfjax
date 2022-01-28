"""
Prokaryotic auto-regulatory gene network Model.

The base model is:

```
x_mt = x_{m, t-1} + mu_mt dt/m + Sigma_mt^{1/2} dt/m
y_t ~ N( exp(x_{m,mt}), diag(tau^2) )
```

Ito's Lemma is applied to transform 
```
logx_mt = log(x_mt)
```
so `mu_mt` and `Sigma_mt` are transformed accordingly. 

- Model parameters: `theta = (theta0, ... theta7, tau0, ... tau3)`.
- Global constants: `dt` and `n_res`, i.e., `m`.
- State dimensions: `n_state = (n_res, 4)`.
- Measurement dimensions: `n_meas = 4`.

"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import lax
from pfjax import sde as sde

# --- main functions -----------------------------------------------------------
class PGNETModel(object):
    def __init__(self, dt, n_res):
        self.dt = dt
        self.n_res = n_res
        self.n_state = (self.n_res, 4)
        self.K = 10

    def premu(self, x, theta, K):
        """
        Calculates pre-transformed `mu` required for the drift function.
        """
        mu1 = theta[2]*x[3] - theta[6]*x[0]
        mu2 = 2*theta[5]*x[2] - theta[7]*x[1] + theta[3]*x[0] - theta[4]*x[1]*(x[1]-1)
        mu3 = theta[1]*(K-x[3]) - theta[0]*x[3]*x[2] - theta[5]*x[2] + 0.5*theta[4]*x[1]*(x[1]-1)
        mu4 = theta[1]*(K-x[3]) - theta[0]*x[3]*x[2]
        mu = jnp.stack([mu1, mu2, mu3, mu4])
        return mu

    def preSigma(self, x, theta, K):
        """
        Calculates pre-transformed `Sigma` required for the drift and diff function.
        """
        A = theta[0]*x[3]*x[2] + theta[1]*(K-x[3])
        sigma11 = theta[2]*x[3] + theta[6]*x[0]
        sigma_max = jnp.where(0 < x[1]*(x[1]-1), x[1]*(x[1]-1), 0)
        sigma_max = x[1]*(x[1]-1)
        sigma22 = theta[7]*x[1] + 4*theta[5]*x[2] + theta[3]*x[0] + 2*theta[4]*sigma_max
        sigma23 = -2*theta[5]*x[2] - theta[4]*sigma_max
        sigma33 = A + theta[5]*x[2] + 0.5*theta[4]*sigma_max
        sigma34 = A
        sigma44 = A

        Sigma = jnp.array([[sigma11, 0, 0, 0],
                           [0, sigma22, sigma23, 0],
                           [0, sigma23, sigma33, sigma34],
                           [0, 0, sigma34, sigma44]])
        
        return Sigma

    def drift(self, x, theta):
        """
        Calculates the SDE drift function.
        """
        x = jnp.exp(x)
        K = self.K
        mu = self.premu(x, theta, K)
        Sigma  = self.preSigma(x, theta, K)
        
        #f_p = jnp.array([1/x[0], 1/x[1], 1/x[2], 1/x[3] + 1/(K-x[3])])
        #f_pp = jnp.array([-1/x[0]/x[0], -1/x[1]/x[1], -1/x[2]/x[2], -1/x[3]/x[3] + 1/(K-x[3])/(K-x[3])])
        f_p = jnp.array([1/x[0], 1/x[1], 1/x[2], 1/x[3]])
        f_pp = jnp.array([-1/x[0]/x[0], -1/x[1]/x[1], -1/x[2]/x[2], -1/x[3]/x[3]])
        
        mu_trans = f_p * mu + 0.5 * f_pp * jnp.diag(Sigma)
        return mu_trans

    def diff(self, x, theta):
        """
        Calculates the SDE diffusion function.
        """
        x = jnp.exp(x)
        K = self.K
        Sigma = self.preSigma(x, theta, K)

        #f_p = jnp.array([1/x[0], 1/x[1], 1/x[2], 1/x[3] + 1/(K-x[3])])
        f_p = jnp.array([1/x[0], 1/x[1], 1/x[2], 1/x[3]])
        Sigma_trans = jnp.outer(f_p, f_p) * Sigma

        return Sigma_trans

    # def drift(self, x, theta):
    #     K = self.K
    #     mu = self.premu(x, theta, K)
    #     return mu

    # def diff(self, x, theta):
    #     K = self.K
    #     Sigma = self.preSigma(x, theta, K)
    #     return Sigma
        
    def state_lpdf(self, x_curr, x_prev, theta):
        """
        Calculates the log-density of `p(x_curr | x_prev, theta)`.

        Args:
            x_curr: State variable at current time `t`.
            x_prev: State variable at previous time `t-1`.
            theta: Parameter value.

        Returns:
            The log-density of `p(x_curr | x_prev, theta)`.
        """
        x = jnp.append(jnp.expand_dims(x_prev[self._n_res-1], axis=0),
                       x_curr, axis=0)
        return sde.euler_lpdf_var(x, self.dt/self._n_res,
                                  self.drift, self.diff, theta)

    def state_sample(self, key, x_prev, theta):
        """
        Samples from `x_curr ~ p(x_curr | x_prev, theta)`.

        Args:
            x_prev: State variable at previous time `t-1`.
            theta: Parameter value.
            key: PRNG key.

        Returns:
            Sample of the state variable at current time `t`: `x_curr ~ p(x_curr | x_prev, theta)`.
        """
        return sde.euler_sim_var(
            n_steps=self.n_res,
            x=x_prev[self.n_res-1],
            dt=self.dt/self.n_res,
            drift=self.drift,
            diff=self.diff,
            theta=theta,
            key=key
        )

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
        tau = theta[8:12]
        return jnp.sum(
            jsp.stats.norm.logpdf(y_curr, loc=jnp.exp(x_curr[-1]), scale=tau)
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
        tau = theta[8:12]
        return jnp.exp(x_curr[-1]) + tau * random.normal(key, (self._n_state[1],))

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

        **FIxME:** Explain what the proposal is and why it gives `logw = 0`.

        In fact, if you think about it hard enough then it's not actually a perfect proposal...

        Args:
            y_init: Measurement variable at initial time `t = 0`.
            theta: Parameter value.
            key: PRNG key.

        Returns:
            - x_init: A sample from the proposal distribution for `x_init`.
            - logw: The log-weight of `x_init`.
        """
        tau = theta[8:12]
        # key, subkey = random.split(key)
        # x_init = jnp.log(y_init + 
        #         tau * random.normal(subkey, (self.n_state[1],)))
        # return \
        #     jnp.append(jnp.zeros((self.n_res-1,) + x_init.shape),
        #                jnp.expand_dims(x_init, axis=0), axis=0), \
        #     jnp.zeros(())

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
