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
from jax import lax, random
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
    return (
        x
        + jnp.array([alpha - beta * jnp.exp(x[1]), -gamma + delta * jnp.exp(x[0])]) * dt
    )


# def euler_sim(key, n_steps, x, dt, theta, n_state):
#     """
#     Euler simulation for `n_steps`.

#     **FIXME:** Put this into an SDE module, possibly with different dispatch depending on whether diffusion is constant, diagonal, etc.
#     """
#     sigma = theta[4:6] * jnp.sqrt(dt)

#     # setup lax.scan:
#     # scan function
#     def fun(carry, t):
#         key, subkey = random.split(carry["key"])
#         x = lotvol_drift(carry["x"], dt, theta) + \
#             random.normal(subkey, (n_state[1],)) * sigma
#         res = {"x": x, "key": key}
#         return res, res
#     # scan initial value
#     init = {"x": x, "key": key}
#     # lax.scan itself
#     last, full = lax.scan(fun, init, jnp.arange(n_steps))
#     return full["x"]


# --- main functions -----------------------------------------------------------


class LotVolModel(object):
    def __init__(self, dt, n_res):
        self.dt = dt
        self.n_res = n_res
        self.n_state = (self.n_res, 2)

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
        return theta[4:6]

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
        x = jnp.append(jnp.expand_dims(x_prev[self.n_res - 1], axis=0), x_curr, axis=0)
        # return sde.euler_lpdf_diag(x, self.dt/self.n_res,
        #                            self.drift, self.diff, theta)
        x0 = jnp.append(
            jnp.expand_dims(x_prev[self.n_res - 1], axis=0),
            x_curr[: self.n_res - 1],
            axis=0,
        )
        x1 = x_curr
        dt_res = self.dt / self.n_res
        sigma = theta[4:6] * jnp.sqrt(dt_res)
        lp = jax.vmap(
            lambda t: jnp.sum(
                jsp.stats.norm.logpdf(
                    x1[t], loc=lotvol_drift(x0[t], dt_res, theta), scale=sigma
                )
            )
        )(jnp.arange(self.n_res))
        return jnp.sum(lp)

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
        dt_res = self.dt / self.n_res
        x0 = jnp.append(
            jnp.expand_dims(x_prev[self.n_res - 1], axis=0),
            x_curr[: self.n_res - 1],
            axis=0,
        )
        x1 = x_curr
        sigma = theta[4:6] * jnp.sqrt(dt_res)
        lp = jnp.array(0.0)
        for t in range(self.n_res):
            lp = lp + jnp.sum(
                jsp.stats.norm.logpdf(
                    x1[t], loc=lotvol_drift(x0[t], dt_res, theta), scale=sigma
                )
            )
        return lp

    def state_sample(self, key, x_prev, theta):
        """
        Samples from `x_curr ~ p(x_curr | x_prev, theta)`.

        Args:
            key: PRNG key.
            x_prev: State variable at previous time `t-1`.
            theta: Parameter value.

        Returns:
            Sample of the state variable at current time `t`: `x_curr ~ p(x_curr | x_prev, theta)`.
        """
        dt_res = self.dt / self.n_res
        sigma = theta[4:6] * jnp.sqrt(dt_res)
        # lax.scan setup:

        # scan function
        def fun(carry, t):
            key, subkey = random.split(carry["key"])
            x = lotvol_drift(carry["x"], dt_res, theta) + sigma * random.normal(
                subkey, (2,)
            )
            res = {"x": x, "key": key}
            return res, res

        # scan initial value
        init = {"x": x_prev[self.n_res - 1], "key": key}
        last, full = lax.scan(fun, init, jnp.arange(self.n_res))
        return full["x"]
        # return sde.euler_sim_diag(
        #     key=key,
        #     n_steps=self.n_res,
        #     x=x_prev[self.n_res-1],
        #     dt=self.dt/self.n_res,
        #     drift=self.drift,
        #     diff=self.diff,
        #     theta=theta
        # )
        # return euler_sim(self.n_res, x_prev[self.n_res-1], self.dt/self.n_res, theta, key, self.n_state)

    def state_sample_for(self, key, x_prev, theta):
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
        dt_res = self.dt / self.n_res
        sigma = theta[4:6] * jnp.sqrt(dt_res)
        x_curr = jnp.zeros(self.n_state)
        x_state = x_prev[self.n_res - 1]
        for t in range(self.n_res):
            key, subkey = random.split(key)
            x_state = (
                lotvol_drift(x_state, dt_res, theta)
                + random.normal(subkey, (self.n_state[1],)) * sigma
            )
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
        return jnp.exp(x_curr[-1]) + tau * random.normal(key, (self.n_state[1],))

    # def init_logw(self, x_init, y_init, theta):
    #     """
    #     Log-weight of the importance sampler for initial state variable `x_init`.

    #     Suppose that
    #     ```
    #     x_init ~ q(x_init) = q(x_init | y_init, theta)
    #     ```
    #     Then function returns
    #     ```
    #     logw = log p(y_init | x_init, theta) + log p(x_init | theta) - log q(x_init)
    #     ```

    #     Args:
    #         x_init: State variable at initial time `t = 0`.
    #         y_init: Measurement variable at initial time `t = 0`.
    #         theta: Parameter value.

    #     Returns:
    #         The log-weight of the importance sampler for `x_init`.
    #     """
    #     return jnp.zeros(())

    # def init_sample(self, y_init, theta, key):
    #     """
    #     Sampling distribution for initial state variable `x_init`.

    #     Samples from an importance sampling proposal distribution
    #     ```
    #     x_init ~ q(x_init) = q(x_init | y_init, theta)
    #     ```
    #     See `init_logw()` for details.

    #     Args:
    #         y_init: Measurement variable at initial time `t = 0`.
    #         theta: Parameter value.
    #         key: PRNG key.

    #     Returns:
    #         Sample from the proposal distribution for `x_init`.
    #     """
    #     tau = theta[6:8]
    #     key, subkey = random.split(key)
    #     x_init = jnp.log(y_init + tau * random.truncated_normal(
    #         subkey,
    #         lower=-y_init/tau,
    #         upper=jnp.inf,
    #         shape=(self.n_state[1],)
    #     ))
    #     return jnp.append(jnp.zeros((self.n_res-1,) + x_init.shape),
    #                       jnp.expand_dims(x_init, axis=0), axis=0)

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
        key, subkey = random.split(key)
        x_init = jnp.log(
            y_init
            + tau
            * random.truncated_normal(
                subkey, lower=-y_init / tau, upper=jnp.inf, shape=(self.n_state[1],)
            )
        )
        logw = jnp.sum(jsp.stats.norm.logcdf(y_init / tau))
        return (
            jnp.append(
                jnp.zeros((self.n_res - 1,) + x_init.shape),
                jnp.expand_dims(x_init, axis=0),
                axis=0,
            ),
            logw,
        )
        # x_init = jnp.log(y_init +
        #                  tau * random.normal(subkey, (self.n_state[1],)))
        # return \
        #     jnp.append(jnp.zeros((self.n_res-1,) + x_init.shape),
        #                jnp.expand_dims(x_init, axis=0), axis=0), \
        #     jnp.zeros()
        # x_init = jnp.log(y_init +
        #                  tau * random.normal(subkey, (self.n_state[1],)))
        # x_step = euler_sim(self.n_res-1, x_init, self.dt / self.n_res,
        #                    theta, key, self.n_state)
        # return jnp.append(jnp.expand_dims(x_init, axis=0), x_step, axis=0),
        # jnp.zeros(())

    def pf_step(self, key, x_prev, y_curr, theta):
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
        x_curr = self.state_sample(key, x_prev, theta)
        logw = self.meas_lpdf(y_curr, x_curr, theta)
        return x_curr, logw
