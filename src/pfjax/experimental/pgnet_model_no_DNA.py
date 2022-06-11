"""
Prokaryotic auto-regulatory gene network Model with unobserved DNA 

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
y_t ~ N( exp(x_{m,mt}), diag(tau^2) )
```

Ito's Lemma is applied to transform the base model on the log-scale to allow for unconstrained variables
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


class PGNETModelNoDNA(sde.SDEModel):

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
        self._bootstrap = bootstrap

    def _parse_params (self, params):
        """ 
            kappa, tau, dna0 = self._parse_params(theta)
        """
        theta = params[:8]
        tau = params[8:11]
        dna0 = params[11]
        return theta, tau, dna0

    def _drift(self, x, theta):
        """
        Calculate the drift on the original scale.
        """
        mu1 = theta[2]*x[3] - theta[6]*x[0]
        mu2 = 2*theta[5]*x[2] - theta[7]*x[1] + \
            theta[3]*x[0] - theta[4]*x[1]*(x[1]-1)
        mu3 = theta[1]*(self._K-x[3]) - theta[0]*x[3]*x[2] - \
            theta[5]*x[2] + 0.5*theta[4]*x[1]*(x[1]-1)
        mu4 = theta[1]*(self._K-x[3]) - theta[0]*x[3]*x[2]
        mu = jnp.stack([mu1, mu2, mu3, mu4])
        return mu

    def _diff(self, x, theta):
        """
        Calculate the diffusion matrix on the original scale.
        """
        A = theta[0]*x[3]*x[2] + theta[1]*(self._K-x[3])
        sigma11 = theta[2]*x[3] + theta[6]*x[0]
        sigma_max = jnp.where(0 < x[1]*(x[1]-1), x[1]*(x[1]-1), 0)
        sigma_max = x[1]*(x[1]-1)
        sigma22 = theta[7]*x[1] + 4*theta[5]*x[2] + \
            theta[3]*x[0] + 2*theta[4]*sigma_max
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
        Calculates the SDE drift function on the log scale.
        """
        x = jnp.exp(x)
        # K = self._K
        mu = self._drift(x, theta)
        Sigma = self._diff(x, theta)

        #f_p = jnp.array([1/x[0], 1/x[1], 1/x[2], 1/x[3] + 1/(K-x[3])])
        #f_pp = jnp.array([-1/x[0]/x[0], -1/x[1]/x[1], -1/x[2]/x[2], -1/x[3]/x[3] + 1/(K-x[3])/(K-x[3])])
        f_p = jnp.array([1/x[0], 1/x[1], 1/x[2], 1/x[3]])
        f_pp = jnp.array(
            [-1/x[0]/x[0], -1/x[1]/x[1], -1/x[2]/x[2], -1/x[3]/x[3]]
        )

        mu_trans = f_p * mu + 0.5 * f_pp * jnp.diag(Sigma)
        return mu_trans

    def diff(self, x, theta):
        """
        Calculates the SDE diffusion function on the log scale.
        """
        x = jnp.exp(x)
        # K = self._K
        Sigma = self._diff(x, theta)

        #f_p = jnp.array([1/x[0], 1/x[1], 1/x[2], 1/x[3] + 1/(K-x[3])])
        f_p = jnp.array([1/x[0], 1/x[1], 1/x[2], 1/x[3]])
        Sigma_trans = jnp.outer(f_p, f_p) * Sigma

        return Sigma_trans

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
        # tau = theta[8:11]
        _, tau, _ = self._parse_params(theta)
        return jnp.sum(
            jsp.stats.norm.logpdf(y_curr, loc=jnp.exp(x_curr[-1, :3]), scale=tau)
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
        _, tau, _ = self._parse_params(theta)
        return jnp.exp(x_curr[-1, :3]) + tau * random.normal(key, (self._n_state[1]-1,))

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
        _, tau, dna0 = self._parse_params(theta)
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
            shape=(self._n_state[1]-1,)
        ))
        x_init = jnp.append(x_init, jnp.log(dna0))
        logw = jnp.sum(jsp.stats.norm.logcdf(y_init/tau))
        #x_init = theta[12:16]
        #logw = -jnp.float_(0)

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
        _theta, tau, dna0 = self._parse_params(theta)
        if self._bootstrap:
            x_curr, logw = super().pf_step(key, x_prev, y_curr, jnp.append(_theta, tau))
        else:
            omega = (tau / y_curr)**2
            x_curr, logw = self.bridge_prop(
                key, x_prev, y_curr, jnp.append(_theta, tau), 
                jnp.log(y_curr), jnp.eye(4)[:-1, :], jnp.diag(omega)
            )
        return x_curr, logw

