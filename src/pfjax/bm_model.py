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
        self.n_state = 1
        self.n_meas = 1
        self.dt = dt

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
        mu = theta[0]
        sigma = theta[1]
        return jnp.squeeze(
            jsp.stats.norm.logpdf(x_curr, loc=x_prev + mu * self.dt,
                                  scale=sigma * jnp.sqrt(self.dt))
        )

    def state_sample(self, x_prev, theta, key):
        """
        Samples from `x_curr ~ p(x_curr | x_prev, theta)`.

        Args:
            x_prev: State variable at previous time `t-1`.
            theta: Parameter value.
            key: PRNG key.

        Returns:
            Sample of the state variable at current time `t`: `x_curr ~ p(x_curr | x_prev, theta)`.
        """
        mu = theta[0]
        sigma = theta[1]
        x_mean = x_prev + mu * self.dt
        x_sd = sigma * jnp.sqrt(self.dt)
        return x_mean + x_sd * random.normal(key=key)

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
        tau = theta[2]
        return jnp.squeeze(
            jsp.stats.norm.logpdf(y_curr, loc=x_curr, scale=tau)
        )

    def meas_sample(self, x_curr, theta, key):
        """
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
        """
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

    def init_sample(self, y_init, theta, key):
        """
        Sampling distribution for initial state variable `x_init`. 

        Samples from an importance sampling proposal distribution
        ```
        x_init ~ q(x_init) = q(x_init | y_init, theta)
        ```
        See `init_logw()` for details.

        Args:
            y_init: Measurement variable at initial time `t = 0`.
            theta: Parameter value.
            key: PRNG key.

        Returns:
            Sample from the proposal distribution for `x_init`.
        """
        return self.meas_sample(y_init, theta, key)

    def pf_init(self, y_init, theta, key):
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

        Args:
            y_init: Measurement variable at initial time `t = 0`.
            theta: Parameter value.
            key: PRNG key.

        Returns:
            - x_init: A sample from the proposal distribution for `x_init`.
            - logw: The log-weight of `x_init`.
        """
        return self.meas_sample(y_init, theta, key), jnp.zeros(())

    def pf_step(self, x_prev, y_curr, theta, key):
        """
        Particle filter calculation for `x_curr`. 

        Samples from an importance sampling proposal distribution
        ```
        x_curr ~ q(x_curr) = q(x_curr | x_prev, y_curr, theta)
        ```
        and calculates the log weight
        ```
        logw = log p(y_curr | x_curr, theta) + log p(x_curr | x_prev, theta) - log q(x_curr)
        ```

        **FIXME:** Explain that this is a bootstrap particle filter.

        Args:
            x_prev: State variable at previous time `t-1`.
            y_curr: Measurement variable at current time `t`.
            theta: Parameter value.
            key: PRNG key.

        Returns:
            - x_curr: Sample of the state variable at current time `t`: `x_curr ~ q(x_curr)`.
            - logw: The log-weight of `x_curr`.
        """
        x_curr = self.state_sample(x_prev, theta, key)
        logw = self.meas_lpdf(y_curr, x_curr, theta)
        return x_curr, logw
