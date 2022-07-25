"""
Prokaryotic auto-regulatory gene network Model.

The base model involves differential equations of the chemical reactions:

```
DNA + P2 --> DNA_P2
DNA_P2   --> DNA + P2
DNA      --> DNA + RNA
RNA      --> RNA + P
P + P    --> P2
P2       --> P + P
RNA      --> 0
P        --> 0
```
These equations are associated with a parameter in `theta = (theta0, ..., theta7)`.
The model is approximated by a SDE described in Golightly & Wilkinson (2005). 
A particular restriction on the chemical reactions is by the conservation law which implies that `DNA + DNA_P2 = K`.
Thus the SDE model can be described in terms of `x_t = (RNA, P, P2, DNA)`. 

Then assuming a standard form of the SDE, the base model can be written as
```
x_mt = x_{m, t-1} + mu_mt dt/m + Sigma_mt^{1/2} dt/m
y_t ~ N( x_{m,mt}, diag(tau^2) )
```

This model is on the regular scale.


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


class RegPGNETModel(sde.SDEModel):

    def __init__(self, dt, n_res, bootstrap=True):
        r"""
        Class constructor for the PGNET model.

        Args:
            dt: SDE interobservation time.
            n_res: SDE resolution number.  There are `n_res` latent variables per observation, equally spaced with interobservation time `dt/n_res`.
            bootstrap (bool): Flag indicating whether to use a Bootstrap particle filter or a bridge filter.

        """
        # creates "private" variables self._dt and self._n_res
        super().__init__(dt, n_res, diff_diag=False)
        self._n_state = (self._n_res, 4)
        self._K = 10
        self._eps = 1e-10
        self._bootstrap = bootstrap

    def drift(self, x, theta):
        """
        Calculate the drift on the original scale.
        """
        mu1 = theta[2]*x[3] - theta[6]*x[0]
        sigma_max = jnp.where(0 < x[1]*(x[1]-1), x[1]*(x[1]-1), 0)
        # sigma_max = x[1]*(x[1]-1)
        mu2 = 2*theta[5]*x[2] - theta[7]*x[1] + \
            theta[3]*x[0] - theta[4]*sigma_max 
        mu3 = theta[1]*(self._K-x[3]) - theta[0]*x[3]*x[2] - \
            theta[5]*x[2] + 0.5*theta[4]*sigma_max
        mu4 = theta[1]*(self._K-x[3]) - theta[0]*x[3]*x[2]
        mu = jnp.stack([mu1, mu2, mu3, mu4])
        return mu

    def diff(self, x, theta):
        """
        Calculate the diffusion matrix on the original scale.
        """
        A = theta[0]*x[3]*x[2] + theta[1]*(self._K-x[3])
        sigma11 = theta[2]*x[3] + theta[6]*x[0]
        sigma_max = jnp.where(0 < x[1]*(x[1]-1), x[1]*(x[1]-1), 0)
        # sigma_max = x[1]*(x[1]-1)
        sigma22 = theta[7]*x[1] + 4*theta[5]*x[2] + \
            theta[3]*x[0] + 2*theta[4]*sigma_max
        sigma23 = -2*theta[5]*x[2] - theta[4]*sigma_max
        sigma33 = A + theta[5]*x[2] + 0.5*theta[4]*sigma_max
        sigma34 = A
        sigma44 = A

        Sigma = jnp.array([[sigma11, 0., 0., 0.],
                           [0., sigma22, sigma23, 0.],
                           [0., sigma23, sigma33, sigma34],
                           [0., 0, sigma34, sigma44]])

        return Sigma

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
            jsp.stats.norm.logpdf(y_curr, loc=x_curr[-1], scale=tau)
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
        return x_curr[-1] + tau * random.normal(key, (self._n_state[1],))

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
        tau = theta[8:12]
        # key, subkey = random.split(key)
        # x_init = jnp.log(y_init +
        #         tau * random.normal(subkey, (self.n_state[1],)))
        # return \
        #     jnp.append(jnp.zeros((self.n_res-1,) + x_init.shape),
        #                jnp.expand_dims(x_init, axis=0), axis=0), \
        #     jnp.zeros(())

        key, subkey = random.split(key)
        x_init123 = y_init[:3] + tau[:3] * random.truncated_normal(
            subkey,
            lower=-y_init[:3]/tau[:3],
            upper=jnp.inf,
            shape=(self._n_state[1]-1,)
        )

        x_init4 = y_init[3] + tau[3] * random.truncated_normal(
            subkey,
            lower=-y_init[3]/tau[3],
            upper=(self._K - y_init[3])/tau[3],
            shape=(1,)
        )
        x_init = jnp.append(x_init123, x_init4)
        logw = jnp.sum(jsp.stats.norm.logcdf(y_init/tau))

        return \
            jnp.append(jnp.zeros((self._n_res-1,) + x_init.shape),
                       jnp.expand_dims(x_init, axis=0), axis=0), \
            logw

    def pf_step(self, key, x_prev, y_curr, theta):
        """
        Choose between bootstrap filter and bridge proposal.

        Args:
            x_prev: State variable at previous time `t-1`.
            y_curr: Measurement variable at current time `t`.
            theta: Parameter value.
            key: PRNG key.

        Returns:
            - x_curr: Sample of the state variable at current time `t`: `x_curr ~ q(x_curr)`.
            - logw: The log-weight of `x_curr`.
        """
        if self._bootstrap:
            x_curr, logw = super().pf_step(key, x_prev, y_curr, theta)
        else:
            omega = theta[8:12]**2
            
            x_curr, logw = self.bridge_prop(
                key, x_prev, y_curr, theta, 
                y_curr, jnp.eye(4), jnp.diag(omega)
            )
        return x_curr, logw

    def is_valid(self, x, theta):
        """
        Checks whether SDE observations are valid.

        Args:
            x: SDE variables.  A vector of size `n_dims`.
            theta: Parameter value.

        Returns:
            Whether or not `x>=0`.
        """
        return (x >= 0) & (x[3] <= self._K)
