"""
Generic methods for SDEs.

**Notes:**

- `euler_{sim/lpdf}_{diag/var}()` are designed to be used independently of the SDE base class.  If we abandon this requirement then we can avoid quite a bit of code duplication by having these functions do one data point only and putting the `vmap`/`scan` constructs into the `state_{sample/lpdf}` methods.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import lax


def euler_sim_diag(key, n_steps, x, dt, drift, diff, theta):
    """
    Simulate SDE with diagonal diffusion using Euler-Maruyama discretization.

    Args:
        key: PRNG key.
        n_steps: Number of steps to simulate.
        x: Initial value of the SDE.  A vector of size `n_dims`.
        dt: Interobservation time.
        drift: Drift function having signature `drift(x, theta)` and returning a vector of size `n_dims`.
        diff: Diffusion function having signature `diff(x, theta)` and returning a vector of size `n_dims`.
        theta: Parameter value.

    Returns:
        Simulated SDE values in a matrix of size `n_steps x n_dims`.
    """

    # setup lax.scan:
    # scan function
    def fun(carry, t):
        key, subkey = random.split(carry["key"])
        x = carry["x"]
        dr = x + drift(x, theta) * dt
        df = diff(x, theta) * jnp.sqrt(dt)
        x = dr + df * random.normal(subkey, (x.shape[0],))
        res = {"x": x, "key": key}
        return res, res
    # scan initial value
    init = {"x": x, "key": key}
    # lax.scan itself
    last, full = lax.scan(fun, init, jnp.arange(n_steps))
    return full["x"]


def euler_lpdf_diag(x, dt, drift, diff, theta):
    """
    Calculate the log PDF of observations from an SDE with diagonal diffusion using the Euler-Maruyama discretization.

    Args:
        x: SDE observations.  An array of size `n_obs x n_dims`.
        dt: Interobservation time.
        drift: Drift function having signature `drift(x, theta)` and returning a vector of size `n_dims`.
        diff: Diffusion function having signature `diff(x, theta)` and returning a vector of size `n_dims`.
        theta: Parameter value.

    Returns:
        The log-density of the SDE observations.
    """
    x0 = x[:-1, :]
    x1 = x[1:, :]
    lp = jax.vmap(lambda t:
                  jsp.stats.norm.logpdf(
                      x=x1[t],
                      loc=x0[t] + drift(x0[t], theta) * dt,
                      scale=diff(x0[t], theta) * jnp.sqrt(dt)
                  ))(jnp.arange(x0.shape[0]))
    return jnp.sum(lp)


class SDEModel(object):
    """
    Base class for SDE models.

    This class should set up a PF model class with methods `state_lpdf()`, `state_sim()`, and `pf_step()`,  with the user only needing to specify SDE drift and diffusion functions, and whether the diffusion is on the `diag` scale.

    For the latter, methods `euler_sim()` and `euler_lpdf()` are supplied at instantiation time from either `euler_{sim/lpdf}_diag()` or `euler_{sim/lpdf}_var()`, with arguments identical to those of the free functions except `drift` and `diff`, which are supplied by `self.drift()` and `self.diff()`.  Hopefully this won't be a problem when we come to jitting, gradding, etc.

    For `pf_step()`, a bootstrap filter is assumed, for which the user needs to specify `meas_lpdf()`.

    **Notes:**

    - Currently contains `state_sample_for()` and `state_lpdf_for()` for testing purposes.  May want to move these elsewhere at some point to obfuscate from users...
    """

    def __init__(self, dt, n_res, diff_diag):
        """
        Class constructor.

        Args:
            dt: SDE interobservation time.
            n_res: SDE resolution number.  There are `n_res` latent variables per observation, equally spaced with interobservation time `dt/n_res`.
            diff_diag: Whether or not the diffusion matrix is assumed to be diagonal.
        """
        self._dt = dt
        self._n_res = n_res
        self._diff_diag = diff_diag  # currently only used for testing
        # instantiate methods euler_sim and euler_lpdf from free functions
        if diff_diag:
            def euler_sim(self, key, n_steps, x, dt, theta):
                return euler_sim_diag(key, n_steps, x, dt,
                                      self.drift, self.diff, theta)

            def euler_lpdf(self, x, dt, theta):
                return euler_lpdf_diag(x, dt, self.drift, self.diff, theta)

            setattr(self.__class__, 'euler_sim', euler_sim)
            setattr(self.__class__, 'euler_lpdf', euler_lpdf)
        else:
            def euler_sim(self, key, n_steps, x, dt, theta):
                return euler_sim_var(key, n_steps, x, dt,
                                     self.drift, self.diff, theta)

            def euler_lpdf(self, x, dt, theta):
                return euler_lpdf_var(x, dt, self.drift, self.diff, theta)

            setattr(self.__class__, 'euler_sim', euler_sim)
            setattr(self.__class__, 'euler_lpdf', euler_lpdf)

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
        return self.euler_lpdf(x, self._dt/self._n_res, theta)

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
        dt_res = self.dt/self.n_res
        x0 = jnp.append(jnp.expand_dims(
            x_prev[self.n_res-1], axis=0), x_curr[:self.n_res-1], axis=0)
        x1 = x_curr
        lp = jnp.array(0.0)
        for t in range(self.n_res):
            if self._diff_diag:
                lp = lp + jnp.sum(jsp.stats.norm.logpdf(
                    x=x1[t],
                    loc=x0[t] + drift(x0[t], theta) * dt_res,
                    scale=diff(x0[t], theta) * jnp.sqrt(dt_res)
                ))
            else:
                lp = lp + jnp.sum(jsp.stats.multivariate_normal.logpdf(
                    x=x1[t],
                    loc=x0[t] + drift(x0[t], theta) * dt_res,
                    scale=diff(x0[t], theta) * jnp.sqrt(dt_res)
                ))
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
        return self.euler_sim(
            key=key,
            n_steps=self._n_res,
            x=x_prev[self._n_res-1],
            dt=self._dt/self._n_res,
            theta=theta
        )

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
        dt_res = self.dt/self.n_res
        x_curr = jnp.zeros(self.n_state)
        x_state = x_prev[self.n_res-1]
        for t in range(self.n_res):
            key, subkey = random.split(key)
            if self._diff_diag:
                dr = x_state + drift(x_state, theta) * dt_res
                df = diff(x_state, theta) * jnp.sqrt(dt_res)
                x_state = dr + df * random.normal(subkey, (x_state.shape[0],))
            else:
                dr = x_state + drift(x_state, theta) * dt_res
                df = diff(x_state, theta) * dt_res
                x_state = random.multivariate_normal(subkey, mean=dr, cov=df)
            x_curr = x_curr.at[t].set(x_state)
        return x_curr

    def pf_step(self, key, x_prev, y_curr, theta):
        """
        Update particle and calculate log-weight for a bootstrap particle filter.

        **FIXME:** This method is completely generic, i.e., is not specific to SDEs.  May wish to put it elsewhere...

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
