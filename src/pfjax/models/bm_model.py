"""
Brownian motion state space model.

The model is:

```
x_0 ~ pi(x_0) \propto 1
x_t ~ N(x_{t-1} + mu * dt, sigma * sqrt(dt))
y_t ~ N(x_t, tau)
```

The parameter values are `theta = (mu, sigma, tau)`, the meaurement and state dimensions are `n_meas = 1` and `n_state = 1`, and `dt` is a (non-static) class member.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random


class BMModel:
    def __init__(self, dt):
        self.n_state = ()
        self.n_meas = ()
        self._dt = dt

    def state_lpdf(self, x_curr, x_prev, theta):
        r"""
        Calculates the log-density of `p(x_curr | x_prev, theta)`.

        Args:
            x_curr: State variable at current time `t`.
            x_prev: State variable at previous time `t-1`.
            theta: Parameter value.

        Returns:
            The log-density of `p(x_curr | x_prev, theta)`.
        """
        mu = theta[0]
        sigma = theta[1]
        return jnp.squeeze(
            jsp.stats.norm.logpdf(x_curr, loc=x_prev + mu * self._dt,
                                  scale=sigma * jnp.sqrt(self._dt))
        )

    def state_sample(self, key, x_prev, theta):
        r"""
        Samples from `x_curr ~ p(x_curr | x_prev, theta)`.

        Args:
            key: PRNG key.
            x_prev: State variable at previous time `t-1`.
            theta: Parameter value.

        Returns:
            Sample of the state variable at current time `t`: `x_curr ~ p(x_curr | x_prev, theta)`.
        """
        mu = theta[0]
        sigma = theta[1]
        x_mean = x_prev + mu * self._dt
        x_sd = sigma * jnp.sqrt(self._dt)
        return x_mean + x_sd * random.normal(key=key)

    def meas_lpdf(self, y_curr, x_curr, theta):
        r"""
        Log-density of `p(y_curr | x_curr, theta)`.

        Args:
            y_curr: Measurement variable at current time `t`.
            x_curr: State variable at current time `t`.
            theta: Parameter value.

        Returns:
            The log-density of `p(y_curr | x_curr, theta)`.
        """
        tau = theta[2]
        return jnp.squeeze(
            jsp.stats.norm.logpdf(y_curr, loc=x_curr, scale=tau)
        )

    def meas_sample(self, key, x_curr, theta):
        r"""
        Sample from `p(y_curr | x_curr, theta)`.

        Args:
            x_curr: State variable at current time `t`.
            theta: Parameter value.
            key: PRNG key.

        Returns:
            Sample of the measurement variable at current time `t`: `y_curr ~ p(y_curr | x_curr, theta)`.
        """
        tau = theta[2]
        return x_curr + tau * random.normal(key=key)

    def init_logw(self, x_init, y_init, theta):
        r"""
        Log-weight of the importance sampler for initial state variable `x_init`.

        Suppose that 
        ```
        x_init ~ q(x_init) = q(x_init | y_init, theta)
        ```
        Then function returns
        ```
        logw = log p(y_init | x_init, theta) + log p(x_init | theta) - log q(x_init)
        ```

        Args:
            x_init: State variable at initial time `t = 0`.
            y_init: Measurement variable at initial time `t = 0`.
            theta: Parameter value.

        Returns:
            The log-weight of the importance sampler for `x_init`.
        """
        # return -meas_lpdf(x_init, y_init, theta)
        return jnp.zeros(())

    def init_sample(self, key, y_init, theta):
        r"""
        Sampling distribution for initial state variable `x_init`. 

        Samples from an importance sampling proposal distribution
        ```
        x_init ~ q(x_init) = q(x_init | y_init, theta)
        ```
        See `init_logw()` for details.

        Args:
            key: PRNG key.
            y_init: Measurement variable at initial time `t = 0`.
            theta: Parameter value.

        Returns:
            Sample from the proposal distribution for `x_init`.
        """
        return self.meas_sample(key, y_init, theta)

    def pf_init(self, key, y_init, theta):
        r"""
        Get particles for initial state `x_init = x_state[0]`. 

        Samples from an importance sampling proposal distribution
        ```
        x_init ~ q(x_init) = q(x_init | y_init, theta)
        ```
        and calculates the log weight
        ```
        logw = log p(y_init | x_init, theta) + log p(x_init | theta) - log q(x_init)
        ```
        In this case we have an exact proposal 
        ```
        q(x_init) = p(x_init | y_init, theta)   <=>   x_init ~ N(y_init, tau)
        ```
        Moreover, due to symmetry of arguments we have `q(x_init) = p(y_init | x_init, theta)`, and since `p(x_init | theta) \propto 1` we have `logw = 0`.

        Args:
            key: PRNG key.
            y_init: Measurement variable at initial time `t = 0`.
            theta: Parameter value.

        Returns:
            - x_init: A sample from the proposal distribution for `x_init`.
            - logw: The log-weight of `x_init`.
        """
        return self.meas_sample(key, y_init, theta), jnp.zeros(())

    def pf_step(self, key, x_prev, y_curr, theta):
        r"""
        Get particles at current step `t_curr` from previous step `t_prev`. 

        Samples from an importance sampling proposal distribution
        ```
        x_curr ~ q(x_curr) = q(x_curr | x_prev, y_curr, theta)
        ```
        and calculates the log weight
        ```
        logw = log p(y_curr | x_curr, theta) + log p(x_curr | x_prev, theta) - log q(x_curr)
        ```

        In this case we have a bootstrap particle filter, so `q(x_curr) = p(x_curr | x_prev, theta)` and `logw = log p(y_curr | x_curr, theta)`.

        Args:
            key: PRNG key.
            x_prev: State variable at previous time `t-1`.
            y_curr: Measurement variable at current time `t`.
            theta: Parameter value.

        Returns:
            - x_curr: Sample of the state variable at current time `t`: `x_curr ~ q(x_curr)`.
            - logw: The log-weight of `x_curr`.
        """
        x_curr = self.state_sample(key, x_prev, theta)
        logw = self.meas_lpdf(y_curr, x_curr, theta)
        return x_curr, logw

    def prop_lpdf(self, x_curr, x_prev, y_curr, theta):
        r"""
        Calculate the log-density of the proposal distribution `q(x_curr) = q(x_curr | x_prev, y_curr, theta)`.

        In this case we have a bootstrap filter, so `q(x_curr) = p(x_curr | x_prev, theta)`.

        Args:
            x_curr: State variable at current time `t`.
            x_prev: State variable at previous time `t-1`.
            y_curr: Measurement variable at current time `t`.
            theta: Parameter value.
        """
        return self.state_lpdf(x_curr=x_curr, x_prev=x_prev, theta=theta)

    def loglik_exact(self, y_meas, theta):
        r"""
        Marginal loglikelihood of the BM model.

        Actually calculates `log p(y_{1:N} | theta, y_0)`, since for the flat prior on `x_0` the marginal likelihood `p(y_{0:N} | theta)` does not exist.

        Args:
            y_meas: Vector of observations `y_0, ..., y_N`.
            theta: Parameter value.

        Returns:
            The marginal loglikelihood `log p(y_{1:N} | theta, y_0)`.
        """
        mu = theta[0]
        sigma2 = theta[1] * theta[1]
        tau2 = theta[2] * theta[2]
        n_obs = y_meas.shape[0]-1  # conditioning on y_0
        t_meas = jnp.arange(1, n_obs+1) * self._dt
        Sigma_y = sigma2 * jax.vmap(lambda t:
                                    jnp.minimum(t, t_meas))(t_meas) + \
            tau2 * (jnp.ones((n_obs, n_obs)) + jnp.eye(n_obs))
        mu_y = y_meas[0] + mu * t_meas
        return jsp.stats.multivariate_normal.logpdf(
            x=jnp.squeeze(y_meas[1:]),
            mean=mu_y,
            cov=Sigma_y
        )
