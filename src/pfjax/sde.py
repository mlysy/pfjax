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
from pfjax import mvn_bridge as mb
from pfjax.utils import *


def euler_sim_diag(key, x, dt, drift, diff, theta):
    """
    Simulate SDE with diagonal diffusion using Euler-Maruyama discretization.

    Args:
        key: PRNG key.
        x: Initial value of the SDE.  A vector of size `n_dims`.
        dt: Interobservation time.
        drift: Drift function having signature `drift(x, theta)` and returning a vector of size `n_dims`.
        diff: Diffusion function having signature `diff(x, theta)` and returning a vector of size `n_dims`.
        theta: Parameter value.

    Returns:
        Simulated SDE values. A vector of size `n_dims`.
    """

    dr = x + drift(x, theta) * dt
    df = diff(x, theta) * jnp.sqrt(dt)
    return dr + df * random.normal(key, (x.shape[0],))


def euler_lpdf_diag(x_curr, x_prev, dt, drift, diff, theta):
    """
    Calculate the log PDF of observations from an SDE with diagonal diffusion using the Euler-Maruyama discretization.

    Args:
        x_curr: Current SDE observations.  A vector of size `n_dims`.
        x_prev: Previous SDE observations.  A vector of size `n_dims`.
        dt: Interobservation time.
        drift: Drift function having signature `drift(x, theta)` and returning a vector of size `n_dims`.
        diff: Diffusion function having signature `diff(x, theta)` and returning a vector of size `n_dims`.
        theta: Parameter value.

    Returns:
        The log-density of the SDE observations.
    """
    return jsp.stats.norm.logpdf(
        x=x_curr,
        loc=x_prev + drift(x_prev, theta) * dt,
        scale=diff(x_prev, theta) * jnp.sqrt(dt)
    )


def euler_sim_var(key, x, dt, drift, diff, theta):
    """
    Simulate SDE with dense diffusion using Euler-Maruyama discretization.

    Args:
        key: PRNG key.
        x: Initial value of the SDE.  A vector of size `n_dims`.
        dt: Interobservation time.
        drift: Drift function having signature `drift(x, theta)` and returning a vector of size `n_dims`.
        diff: Diffusion function having signature `diff(x, theta)` and returning a vector of size `n_dims`.
        theta: Parameter value.

    Returns:
        Simulated SDE values. A vector of size `n_dims`.
    """

    # dr = x + drift(x, theta) * dt
    # chol_Sigma = jnp.linalg.cholesky(diff(x, theta))
    # df = chol_Sigma * jnp.sqrt(dt)
    # return dr + jnp.matmul(df, random.normal(key, (x.shape[0],)))
    return jax.random.multivariate_normal(
        key,
        mean=x + drift(x, theta) * dt,
        cov=diff(x, theta) * dt
    )


def euler_lpdf_var(x_curr, x_prev, dt, drift, diff, theta):
    """
    Calculate the log PDF of observations from an SDE with dense diffusion using the Euler-Maruyama discretization.

    Args:
        x_curr: Current SDE observations.  A vector of size `n_dims`.
        x_prev: Previous SDE observations.  A vector of size `n_dims`.
        dt: Interobservation time.
        drift: Drift function having signature `drift(x, theta)` and returning a vector of size `n_dims`.
        diff: Diffusion function having signature `diff(x, theta)` and returning a vector of size `n_dims`.
        theta: Parameter value.

    Returns:
        The log-density of the SDE observations.
    """
    return jsp.stats.multivariate_normal.logpdf(
        x=x_curr,
        mean=x_prev + drift(x_prev, theta) * dt,
        cov=diff(x_prev, theta) * dt
    )


class SDEModel(object):
    """
    Base class for SDE models.

    This class sets up a PF model with methods `state_lpdf()`, `state_sim()`, and `pf_step()` automatically determined from user-specified SDE drift and diffusion functions.

   For computational efficiency, the user can also specify whether or not the diffusion is diagonal.  This will set up methods `euler_sim()` and `euler_lpdf()` supplied at instantiation time from either `euler_{sim/lpdf}_diag()` or `euler_{sim/lpdf}_var()`, with arguments identical to those of the free functions except `drift` and `diff`, which are supplied by `self.drift()` and `self.diff()`.


    For `pf_step()`, a bootstrap filter is assumed by default, for which the user needs to specify `meas_lpdf()`.

    **Notes:**

    - Currently contains `_state_sample_for()` and `_state_lpdf_for()` for testing purposes.  May want to move these elsewhere at some point to obfuscate from users...

    - Should derive from `pf.BaseModel`.
    """

    def __init__(self, dt, n_res, diff_diag):
        r"""
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
            def euler_sim(self, key, x, dt, theta):
                return euler_sim_diag(key, x, dt,
                                      self.drift, self.diff, theta)

            def euler_lpdf(self, x_curr, x_prev, dt, theta):
                return euler_lpdf_diag(x_curr, x_prev, dt,
                                       self.drift, self.diff, theta)

            def diff_full(self, x, theta):
                return jnp.diag(self.diff(x, theta))

            setattr(self.__class__, 'euler_sim', euler_sim)
            setattr(self.__class__, 'euler_lpdf', euler_lpdf)
            setattr(self.__class__, 'diff_full', diff_full)
        else:
            def euler_sim(self, key, x, dt, theta):
                return euler_sim_var(key, x, dt,
                                     self.drift, self.diff, theta)

            def euler_lpdf(self, x_curr, x_prev, dt, theta):
                return euler_lpdf_var(x_curr, x_prev, dt,
                                      self.drift, self.diff, theta)

            def diff_full(self, x, theta):
                return self.diff(x, theta)

            setattr(self.__class__, 'euler_sim', euler_sim)
            setattr(self.__class__, 'euler_lpdf', euler_lpdf)
            setattr(self.__class__, 'diff_full', diff_full)

    def is_valid(self, x, theta):
        """
        Checks whether SDE observations are valid.

        Args:
            x: SDE variables.  A vector of size `n_dims`.
            theta: Parameter value.

        Returns:
            Whether or not `x` is valid SDE data.
        """
        return jnp.array(True)

    def is_valid_state(self, x, theta):
        """
        Checks whether SDE latent variables are valid.

        Applies `is_valid()` to each of the `n_res` SDE variables, and also checks for nans.

        Args:
            x: State variable of size `n_res x n_dims`.
            theta: Parameter value.

        Returns:
            Whether or not all `n_res` latent variables are valid.
        """
        valid_x = jax.vmap(self.is_valid, in_axes=(0, None))(x, theta)
        nan_x = jnp.any(jnp.isnan(x), axis=1)
        return jnp.alltrue(valid_x, where=~nan_x) & jnp.alltrue(~nan_x)

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
        # No need to do tree version since SDE has 2D x_state
        # x0 = tree_append_first(
        #     x=tree_remove_last(x_curr),
        #     first=tree_keep_last(x_prev)
        # )
        x0 = jnp.concatenate([x_prev[-1][None], x_curr[:-1]])
        x1 = x_curr
        lp = jax.vmap(lambda xp, xc:
                      self.euler_lpdf(
                          x_curr=xc, x_prev=xp,
                          dt=self._dt/self._n_res,
                          theta=theta))(x0, x1)
        # x = jnp.append(jnp.expand_dims(x_prev[self._n_res-1], axis=0),
        #                x_curr, axis=0)
        # lp = jax.vmap(lambda t:
        #               self.euler_lpdf(
        #                   x_curr=x[t+1], x_prev=x[t],
        #                   dt=self._dt/self._n_res,
        #                   theta=theta))(jnp.arange(self._n_res))
        return jnp.sum(lp)
        # x = jnp.append(jnp.expand_dims(x_prev[self._n_res-1], axis=0),
        #                x_curr, axis=0)
        # return self.euler_lpdf(x, self._dt/self._n_res, theta)

    def _state_lpdf_for(self, x_curr, x_prev, theta):
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
                    cov=self.diff(x0[t], theta) * dt_res
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
        # lax.scan setup:
        # scan function
        def fun(carry, t):
            key, subkey = random.split(carry["key"])
            x = self.euler_sim(
                key=subkey, x=carry["x"],
                dt=self._dt/self._n_res, theta=theta
            )
            res = {"x": x, "key": key}
            return res, x
        # scan initial value
        # init = {"x": tree_keep_last(x_prev), "key": key}
        init = {"x": x_prev[-1], "key": key}
        last, full = lax.scan(fun, init, jnp.arange(self._n_res))
        return full
        # return self.euler_sim(
        #     key=key,
        #     n_steps=self._n_res,
        #     x=x_prev[self._n_res-1],
        #     dt=self._dt/self._n_res,
        #     theta=theta
        # )

    def _state_sample_for(self, key, x_prev, theta):
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
            key: PRNG key.
            x_prev: State variable at previous time `t-1`.
            y_curr: Measurement variable at current time `t`.
            theta: Parameter value.

        Returns:
            - x_curr: Sample of the state variable at current time `t`: `x_curr ~ q(x_curr)`.
            - logw: The log-weight of `x_curr`.
        """
        x_curr = self.state_sample(key, x_prev, theta)
        logw = lax.cond(
            self.is_valid_state(x_curr, theta),
            lambda _x: self.meas_lpdf(y_curr, x_curr, theta),
            lambda _x: -jnp.inf,
            0.0
        )
        # logw = self.meas_lpdf(y_curr, x_curr, theta)
        return x_curr, logw

    def _bridge_mv(self, x, theta, n, Y, A, Omega):
        r"""
        Mean and variance of bridge proposal specific for SDEs.
        """
        k = self._n_res - n
        dt_res = self._dt / self._n_res
        dr = self.drift(x, theta) * dt_res
        df = self.diff_full(x, theta) * dt_res
        return mb.mvn_bridge_mv(
            mu_W=x + dr,
            Sigma_W=df,
            mu_Y=jnp.matmul(A, x + k*dr),
            AS_W=jnp.matmul(A, df),
            Sigma_Y=k * jnp.linalg.multi_dot([A, df, A.T]) + Omega,
            Y=Y
        )

    def bridge_step(self, key, x_prev, y_curr, theta, Y, A, Omega):
        """
        Update particle and calculate log-weight for a particle filter with MVN bridge proposals.

        **Notes:**

        - The measurement input is `Y` instead of `y_meas`.  The reason is that the proposal can be used with carefully chosen `Y = g(y_meas)` when the measurement error model is not `y_meas ~ N(A x_state, Omega)`.

        - Only implements the general version with arbitrary `A` and `Omega`.  Specific cases can be done more rapidly, but it's not clear where to put these different methods.  Inside the class?  As free functions?

        - Duplicates a lot of code from `mvn_bridge_pars()`, because `Sigma_Y` and `mu_Y` inside the latter can be computed very easily here.

        - Computes the Euler part of the log-weights using `vmap` after the bridge part which used `lax.scan()`.  On one core, it's definitely faster to do both inside `lax.scan()`.  On multiple cores that may or may not be the case, but probably would need quite a few cores see the speed increase.  However, it's unlikely that we'll explicitly parallelize across cores for this, since the parallellization would typically be over particles.

        - The drift and diffusion functions are each calculated twice, once for proposal and once for Euler.  This is somewhat inefficient, but to circumvent this would need to redesign `euler_lpdf()`...

        Args:
            key: PRNG key.
            x_prev: State variable at previous time `t-1`.
            y_curr: Measurement variable at current time `t`.
            theta: Parameter value.

        Returns:
            Tuple:

            - **x_curr** - Sample of the state variable at current time `t`: `x_curr ~ q(x_curr)`.
            - **logw** - The log-weight of `x_curr`.
        """
        # lax.scan setup
        def scan_fun(carry, n):
            key = carry["key"]
            x = carry["x"]
            # calculate mean and variance of bridge proposal
            mu_bridge, Sigma_bridge = self._bridge_mv(
                x=x, theta=theta, n=n,
                Y=Y, A=A, Omega=Omega
            )
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
        logw = logw + self.meas_lpdf(y_curr, x_prop, theta) - last["lp"]
        logw = lax.cond(
            self.is_valid_state(x_prop, theta),
            lambda _x: logw,
            lambda _x: -jnp.inf,
            0.0
        )
        return x_prop, logw

    def _bridge_step_for(self, key, x_prev, y_curr, theta, Y, A, Omega):
        """
        For-loop version of bridge_step() for testing.
        """

        dt_res = self._dt / self._n_res
        x = x_prev[self._n_res - 1]
        x_prop = []
        lp_prop = jnp.array(0.0)
        for t in range(self._n_res):
            k = self._n_res - t
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
            x = random.multivariate_normal(key,
                                           mean=mu_bridge,
                                           cov=Sigma_bridge)
            x_prop.append(x)
            # bridge log-pdf
            lp_prop = lp_prop + jsp.stats.multivariate_normal.logpdf(
                x=x,
                mean=mu_bridge,
                cov=Sigma_bridge
            )
        x_prop = jnp.array(x_prop)
        logw = self.state_lpdf(
            x_curr=x_prop,
            x_prev=x_prev,
            theta=theta
        )
        logw = logw + self.meas_lpdf(y_curr, x_prop, theta) - lp_prop
        return x_prop, logw
