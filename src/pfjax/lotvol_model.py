"""
this is a test pr
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

- The measurement `y_t` corresponds to `x_t = (x_{m,mt}, ..., x_{m,mt+(m-1)})`, i.e., aligns with the first element of `x_t`.  This makes it easier to define the prior.

- The prior is such that `p(x_0 | y_0, theta)` is given by:

    ```
    exp(x_{m0}) ~ N( y_0, diag(tau_H^2, tau_L^2) )
    x_{mt} ~ Euler(x_{m,t-1}, theta)
    ```

"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import lax


# --- helper functions ---------------------------------------------------------

def drift(x, dt, theta):
    """
    Calculates the SDE drift function.
    """
    alpha = theta[0]
    beta = theta[1]
    gamma = theta[2]
    delta = theta[3]
    return x + jnp.array([alpha - beta * jnp.exp(x[1]),
                          -gamma + delta * jnp.exp(x[0])]) * dt


def euler_sim(n_steps, x, dt, theta, key, n_state):
    """
    Euler simulation for `n_steps`.

    **FIXME:** Put this into an SDE module, possibly with different dispatch depending on whether diffusion is constant, diagonal, etc.
    """
    sigma = theta[4:6] * jnp.sqrt(dt)

    # setup lax.scan:
    # scan function
    def fun(carry, t):
        key, subkey = random.split(carry["key"])
        x = drift(carry["x"], dt, theta) + \
            random.normal(subkey, (n_state[1],)) * sigma
        res = {"x": x, "key": key}
        return res, res
    # scan initial value
    init = {"x": x, "key": key}
    # lax.scan itself
    last, full = lax.scan(fun, init, jnp.arange(n_steps))
    return full["x"]


# --- main functions -----------------------------------------------------------

class LotVolModel(object):
    def __init__(self, dt, n_res):
        self.dt = dt
        self.n_res = n_res
        self.n_state = (self.n_res, 2)

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
        dt_res = self.dt/self.n_res
        x0 = jnp.append(jnp.expand_dims(
            x_prev[self.n_res-1], axis=0), x_curr[:self.n_res-1], axis=0)
        x1 = x_curr
        sigma = theta[4:6] * jnp.sqrt(dt_res)
        lp = jax.vmap(lambda t:
                      jsp.stats.norm.logpdf(x1[t],
                                            loc=drift(x0[t], dt_res, theta),
                                            scale=sigma))(jnp.arange(self.n_res))
        return jnp.sum(lp)

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
        return euler_sim(self.n_res, x_prev[self.n_res-1], self.dt/self.n_res, theta, key, self.n_state)

    def state_sample_for(self, x_prev, theta, key):
        """
        Samples from `x_curr ~ p(x_curr | x_prev, theta)`.

        For-loop version for testing.

        Args:
            x_prev: State variable at previous time `t-1`.
            theta: Parameter value.
            key: PRNG key.

        Returns:
            Sample of the state variable at current time `t`: `x_curr ~ p(x_curr | x_prev, theta)`.
        """
        dt_res = self.dt/self.n_res
        sigma = theta[4:6] * jnp.sqrt(dt_res)
        x_curr = jnp.zeros(self.n_state)
        x_state = x_prev[self.n_res-1]
        for t in range(self.n_res):
            key, subkey = random.split(key)
            x_state = drift(x_state, dt_res, theta) + \
                random.normal(subkey, (self.n_state[1],)) * sigma
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
        tau = theta[6:8]
        return jnp.sum(
            jsp.stats.norm.logpdf(y_curr, loc=jnp.exp(x_curr[0]), scale=tau)
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
        tau = theta[6:8]
        return jnp.exp(x_curr[0]) + tau * random.normal(key, (self.n_state[1],))

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
        tau = theta[6:8]
        key, subkey = random.split(key)
        # x_init = jnp.repeat(jnp.expand_dims(
        #     y_init, axis=0), self.n_res, axis=0)
        # x_init = jnp.log(self.meas_sample(jnp.log(x_init), theta, subkey))
        x_init = jnp.log(y_init +
                         tau * random.normal(subkey, (self.n_state[1],)))
        x_step = euler_sim(self.n_res-1, x_init, self.dt / self.n_res,
                           theta, key, self.n_state)
        return jnp.append(jnp.expand_dims(x_init, axis=0), x_step, axis=0)

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
        key, subkey = random.split(key)
        # x_init = jnp.repeat(jnp.expand_dims(
        #     y_init, axis=0), self.n_res, axis=0)
        # x_init = jnp.log(self.meas_sample(jnp.log(x_init), theta, subkey))
        x_init = jnp.log(y_init +
                         tau * random.normal(subkey, (self.n_state[1],)))
        x_step = euler_sim(self.n_res-1, x_init, self.dt / self.n_res,
                           theta, key, self.n_state)
        return jnp.append(jnp.expand_dims(x_init, axis=0), x_step, axis=0), \
            jnp.zeros(())

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
