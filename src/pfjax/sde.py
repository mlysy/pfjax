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


def mvn_bridge_pars(mu_W, Sigma_W, mu_XW, Sigma_XW, Y, A, Omega):
    """
    Calculate the mean and variance of a normal bridge distribution.

    Suppose we have the multivariate normal model

    ```
           W ~ N(mu_W, Sigma_W)
       X | W ~ N(W + mu_XW, Sigma_XW)
    Y | X, W ~ N(AX, Omega)
    ```

    This function returns the mean and variance of `p(W | Y)`.
    """
    mu_Y = jnp.matmul(A, mu_W + mu_XW)
    AS_W = jnp.matmul(A, Sigma_W)
    Sigma_Y = jnp.linalg.multi_dot([A, Sigma_W + Sigma_XW, A.T]) + Omega
    # solve both linear systems simultaneously
    sol = jnp.matmul(AS_W.T, jnp.linalg.solve(
        Sigma_Y, jnp.hstack([jnp.array([Y-mu_Y]).T, AS_W])))
    return mu_W + jnp.squeeze(sol[:, 0]), Sigma_W - sol[:, 1:]


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

def euler_sim_var(key, n_steps, x, dt, drift, diff, theta):
    """
    Simulate SDE with dense diffusion using Euler-Maruyama discretization.

    Args:
        n_steps: Number of steps to simulate.
        x: Initial value of the SDE.  A vector of size `n_dims`.
        dt: Interobservation time.
        drift: Drift function having signature `drift(x, theta)` and returning a vector of size `n_dims`.
        diff: Diffusion function having signature `diff(x, theta)` and returning a vector of size `n_dims x n_dims`.
        theta: Parameter value.
        key: PRNG key.

    Returns:
        Simulated SDE values in a matrix of size `n_steps x n_dims`.
    """

    # setup lax.scan:
    # scan function
    def fun(carry, t):
        key, subkey = random.split(carry["key"])
        x = carry["x"]
        dr = x + drift(x, theta) * dt
        chol_Sigma = jnp.linalg.cholesky(diff(x, theta))
        df = chol_Sigma * jnp.sqrt(dt)
        #x = random.multivariate_normal(subkey, dr, diff(x, theta)*dt)
        x = dr + jnp.matmul(df, random.normal(subkey, (x.shape[0],)))
        res = {"x": x, "key": key}
        return res, res
    # scan initial value
    init = {"x": x, "key": key}
    # lax.scan itself
    last, full = lax.scan(fun, init, jnp.arange(n_steps))
    return full["x"]


def euler_lpdf_var(x, dt, drift, diff, theta):
    """
    Calculate the log PDF of observations from an SDE with dense diffusion using the Euler-Maruyama discretization.

    Args:
        x: SDE observations.  An array of size `n_obs x n_dims`.
        dt: Interobservation time.
        drift: Drift function having signature `drift(x, theta)` and returning a vector of size `n_dims`.
        diff: Diffusion function having signature `diff(x, theta)` and returning a matrix of size `n_dims x n_dims`.
        theta: Parameter value.

    Returns:
        The log-density of the SDE observations.
    """
    x0 = x[:-1, :]
    x1 = x[1:, :]
    lp = jax.vmap(lambda t:
                  jsp.stats.multivariate_normal.logpdf(
                      x=x1[t],
                      mean=x0[t] + drift(x0[t], theta) * dt,
                      cov=diff(x0[t], theta) * dt
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
        # the following members are only used for testing
        self._diff_diag = diff_diag
        # also have self._n_state for testing,
        # defined in the derived model class
        if diff_diag:
            # instantiate methods depending on whether the diffusion is diagonal
            def euler_sim(self, key, n_steps, x, dt, theta):
                return euler_sim_diag(key, n_steps, x, dt,
                                      self.drift, self.diff, theta)

            def euler_lpdf(self, x, dt, theta):
                return euler_lpdf_diag(x, dt, self.drift, self.diff, theta)

            def diff_full(self, x, theta):
                return jnp.diag(self.diff(x, theta))

            setattr(self.__class__, 'euler_sim', euler_sim)
            setattr(self.__class__, 'euler_lpdf', euler_lpdf)
            setattr(self.__class__, 'diff_full', diff_full)
        else:
            def euler_sim(self, key, n_steps, x, dt, theta):
                return euler_sim_var(key, n_steps, x, dt,
                                     self.drift, self.diff, theta)

            def euler_lpdf(self, x, dt, theta):
                return euler_lpdf_var(x, dt, self.drift, self.diff, theta)

            def diff_full(self, x, theta):
                return self.diff(x, theta)

            setattr(self.__class__, 'euler_sim', euler_sim)
            setattr(self.__class__, 'euler_lpdf', euler_lpdf)
            setattr(self.__class__, 'diff_full', diff_full)

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
        dt_res = self._dt/self._n_res
        x0 = jnp.append(jnp.expand_dims(
            x_prev[self._n_res-1], axis=0), x_curr[:self._n_res-1], axis=0)
        x1 = x_curr
        lp = jnp.array(0.0)
        for t in range(self._n_res):
            if self._diff_diag:
                lp = lp + jnp.sum(jsp.stats.norm.logpdf(
                    x=x1[t],
                    loc=x0[t] + self.drift(x0[t], theta) * dt_res,
                    scale=self.diff(x0[t], theta) * jnp.sqrt(dt_res)
                ))
            else:
                lp = lp + jnp.sum(jsp.stats.multivariate_normal.logpdf(
                    x=x1[t],
                    mean=x0[t] + self.drift(x0[t], theta) * dt_res,
                    cov=self.diff(x0[t], theta) * jnp.sqrt(dt_res)
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
        dt_res = self._dt/self._n_res
        # x_curr = jnp.zeros(self._n_state)
        x_curr = []
        x_state = x_prev[self._n_res-1]
        for t in range(self._n_res):
            key, subkey = random.split(key)
            if self._diff_diag:
                dr = x_state + self.drift(x_state, theta) * dt_res
                df = self.diff(x_state, theta) * jnp.sqrt(dt_res)
                x_state = dr + df * random.normal(subkey, (x_state.shape[0],))
            else:
                dr = x_state + self.drift(x_state, theta) * dt_res
                df = self.diff(x_state, theta) * dt_res
                x_state = random.multivariate_normal(subkey, mean=dr, cov=df)
            x_curr.append(x_state)
            # x_curr = x_curr.at[t].set(x_state)
        return jnp.array(x_curr)

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

    def bridge_prop(self, key, x_prev, Y, theta, A, Omega):
        """
        Bridge proposal.

        **Notes:**

        - The measurement input is `Y` instead of `y_meas`.  The reason is that the proposal can be used with carefully chosen `Y = g(y_meas)` when the measurement error model is not `y_meas ~ N(A x_state, Omega)`.

        - Only implements the general version with arbitrary `A` and `Omega`.  Specific cases can be done more rapidly, but it's not clear where to put these different methods.  Inside the class?  As free functions?

        - Duplicates a lot of code from `mvn_bridge_pars()`, because `Sigma_Y` and `mu_Y` inside the latter can be computed very easily here.

        - Computes the Euler part of the log-weights using `vmap` after the bridge part which used `lax.scan()`.  On one core, it's definitely faster to do both inside `lax.scan()`.  On multiple cores that may or may not be the case, but probably would need quite a few cores see the speed increase.  However, it's unlikely that we'll explicitly parallelize across cores for this, since the parallellization would typically be over particles.

        - The drift and diffusion functions are each calculated twice, once for proposal and once for Euler.  This is somewhat inefficient, but to circumvent this would need to redesign `euler_lpdf()`...
        """
        # lax.scan setup
        def scan_fun(carry, n):
            key = carry["key"]
            x = carry["x"]
            # calculate mean and variance of bridge proposal
            k = self._n_res - n
            dt_res = self._dt / self._n_res
            dr = self.drift(x, theta) * dt_res
            df = self.diff_full(x, theta) * dt_res
            mu_W = x + dr
            Sigma_W = df
            mu_Y = jnp.matmul(A, x + k*dr)
            AS_W = jnp.matmul(A, Sigma_W)
            Sigma_Y = k * jnp.linalg.multi_dot([A, df, A.T]) + Omega
            # solve both linear systems simultaneously
            sol = jnp.matmul(AS_W.T, jnp.linalg.solve(
                Sigma_Y, jnp.hstack([jnp.array([Y-mu_Y]).T, AS_W])))
            mu_bridge = mu_W + jnp.squeeze(sol[:, 0])
            Sigma_bridge = Sigma_W - sol[:, 1:]
            # bridge proposal
            key, subkey = random.split(key)
            x_prop = random.multivariate_normal(key,
                                                mean=mu_bridge,
                                                cov=Sigma_bridge)
            # bridge log-pdf
            lp_prop = jsp.stats.multivariate_normal.logpdf(
                x=x_prop,
                mean=mu_bridge,
                cov=Sigma_bridge
            )
            res_carry = {
                "x": x_prop,
                "key": key,
                "lp": carry["lp"] + lp_prop
            }
            res_stack = {"x": x_prop}
            return res_carry, res_stack
        scan_init = {
            "x": x_prev[self._n_res-1],
            "key": key,
            "lp": jnp.array(0.)
        }
        last, full = lax.scan(scan_fun, scan_init, jnp.arange(self._n_res))
        # return last, full
        x_prop = full["x"]  # bridge proposal
        # log-weight for the proposal
        logw = self.state_lpdf(
            x_curr=x_prop,
            x_prev=x_prev,
            theta=theta
        )
        logw = logw + self.meas_lpdf(y_meas, x_curr, theta) - last["lp"]
        return x_prop, logw
