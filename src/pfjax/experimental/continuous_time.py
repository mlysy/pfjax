import jax
import jax.numpy as jnp
from jax import random
from jax import lax
import pfjax.models


class ContinuousTimeModel(pfjax.models.base_model.BaseModel):

    def __init__(self, dt, n_res, meas_linear, bootstrap):
        r"""
        Base class for continuous-time state-space models.

        Notes:

        - Derived class can provide the following methods:

            - `state_dt_{lpdf/sample}()`: Needed to define `state_{lpdf/sample}()`.

            - `step_dt_{lpdf/sample}()`: Needed to define `step_{lpdf/sample}()`.

            - `pf_dt_step()`: Can be used to define `pf_step()` more efficiently.

        - `prior_{lpdf/sample}()`, `init_{lpdf/sample}()` and `pf_init()` can all be defined wrt a single time point instead of all `n_res` timepoints.  This is expected to be done manually by the user for now, i.e., by subsetting `x_init`.  For the future, how to get users to define these automatically?

        - If `meas_linear == True`, then `meas_{lpdf/sample}()` are created dynamically from `mvn_wv()`.  This means they can't be defined the usual way, i.e., as static methods in the class definition.  It is possible to overwrite this behavior using `getattr()` to determine whether static method exists, or throw error on dynamic method creation if it does.  For this last approach, however, care must be taken to get the error to show up in jitted code.  See here: https://github.com/google/jax/issues/4257.

        - Currently set up for `x_state` being a 2D array (including `n_res` dimension).  Should set this up for `x_state` to be a PyTree with leading dimension `n_res`.
        """
        # creates private variable self._bootstrap
        super().__init__(bootstrap=bootstrap)
        # other private variables
        self._dt = dt
        self._n_res = n_res

        if meas_linear:
            def meas_lpdf(self, y_curr, x_curr, theta):
                (wgt_meas, var_meas) = self.meas_wv(theta)
                return mvn_lpdf(
                    x=y_curr,
                    mean=jnp.dot(wgt_meas, x_curr),
                    var=var_meas
                )

            def meas_sample(self, key, x_curr, theta):
                (wgt_meas, var_meas) = self.meas_wv(theta)
                return mvn_sim(
                    key=key,
                    mean=jnp.dot(wgt_meas, x_curr),
                    var=var_meas
                )

            setattr(self.__class__, 'meas_lpdf', meas_lpdf)
            setattr(self.__class__, 'meas_sample', meas_sample)

    def state_lpdf(self, x_curr, x_prev, theta):
        x0 = jnp.concatenate([x_prev[-1][None], x_curr[:-1]])
        x1 = x_curr
        lp = jax.vmap(
            fun=lambda xp, xc:
            self.state_dt_lpdf(
                x_curr=xc,
                x_prev=xp,
                dt=self._dt/self._n_res,
                theta=theta
            ))(x0, x1)
        return jnp.sum(lp)

    def state_sample(self, key, x_prev, theta):
        # lax.scan function
        def fun(carry, t):
            key, subkey = random.split(carry["key"])
            x = self.state_dt_sim(
                key=subkey,
                x_prev=carry["x"],
                dt=self._dt/self._n_res,
                theta=theta
            )
            res = {"x": x, "key": key}
            return res, x
        # lax.scan initial value
        init = {"x": x_prev[-1], "key": key}
        last, full = lax.scan(fun, init, jnp.arange(self._n_res))
        return full

    def step_lpdf(self, x_curr, x_prev, y_curr, theta):
        x0 = jnp.concatenate([x_prev[-1][None], x_curr[:-1]])
        x1 = x_curr
        dt_sim = self._dt/self._n_res
        lp = jax.vmap(
            fun=lambda xp, xc, n:
            self.step_dt_lpdf(
                x_curr=xc,
                x_prev=xp,
                y_next=y_curr,
                dt_prev=dt_sim,
                dt_next=self._dt - (n+1.)*dt_sim,
                theta=theta
            ))(x0, x1, jnp.arange(self._n_res))
        return jnp.sum(lp)

    def _step_sample(self, key, x_prev, y_curr, theta):
        dt_sim = self._dt/self._n_res

        # lax.scan function
        def fun(carry, n):
            key, subkey = random.split(carry["key"])
            x = self.step_dt_sim(
                key=subkey,
                x_prev=carry["x"],
                y_next=y_curr,
                dt_prev=dt_sim,
                dt_next=self._dt - (n+1.)*dt_sim,
                theta=theta
            )
            res = {"x": x, "key": key}
            return res, x
        # lax.scan initial value
        init = {"x": x_prev[-1], "key": key}
        last, full = lax.scan(fun, init, jnp.arange(self._n_res))
        return full

    def _pf_step(self, key, x_prev, y_curr, theta):
        # lax.scan setup
        def fun(carry, n):
            key = carry["key"]
            x = carry["x"]
            key, subkey = random.split(key)
            x_prop, logw = self.pf_dt_step(
                key=subkey,
                x_prev=carry["x"],
                y_next=y_curr,
                dt_prev=dt_sim,
                dt_next=self._dt - (n+1.)*dt_sim,
                theta=theta
            )
            res_carry = {
                "x": x_prop,
                "key": key
            }
            res_stack = {"x": x_prop, "logw": logw}
            return res_carry, res_stack
        init = {
            "x": x_prev[self._n_res-1],
            "key": key
        }
        _, full = lax.scan(fun, init, jnp.arange(self._n_res))
        return full["x"], jnp.sum(full["logw"])

    def step_sample(self, key, x_prev, y_curr, theta):
        if self._bootstrap:
            fun = super().step_sample
        else:
            fun = self._step_sample
        return fun(
            key=key,
            x_prev=x_prev,
            y_curr=y_curr,
            theta=theta
        )

    def pf_step(self, key, x_prev, y_curr, theta):
        if self._bootstrap:
            fun = super().pf_step
        else:
            fun = self._pf_step
        return fun(
            key=key,
            x_prev=x_prev,
            y_curr=y_curr,
            theta=theta
        )
