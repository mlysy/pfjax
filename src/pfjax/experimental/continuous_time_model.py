import jax
import jax.numpy as jnp
import pfjax.experimental.base_model

# from jax import lax, random, tree


class ContinuousTimeModel(pfjax.experimental.base_model.BaseModel):
    r"""
    Base class for continuous-time state-space models.

    Notes
    -----
    - [x] Derived class can provide the following methods:

        - `state_dt_{lpdf/sample}()`: Needed to define `state_{lpdf/sample}()`.

        - `step_dt_{lpdf/sample}()`: Needed to define `step_{lpdf/sample}()`.

        - `pf_step_dt()`: Can be used to define `pf_step()` more efficiently.

    - [x] `bootstrap == True` argument uses `super()` method to define `step_{lpdf/sample}()` and `pf_step()`, whereas `bootstrap == False` constructs `pf_step_dt()` from `step_dt_{lpdf/sample}()` and `state_dt_lpdf()`.

    - [x] Currently set up for `x_state` being a 2D array (including `n_res` dimension).  Should set this up for `x_state` to be a PyTree with leading dimension `n_res`.

    - [x] `prior_{lpdf/sample}()`, `init_{lpdf/sample}()` and `pf_init()` can all be defined wrt a single time point instead of all `n_res` timepoints.  This is expected to be done manually by the user for now, i.e., by subsetting `x_init`.  For the future, how to get users to define these automatically?

        For now we provide a method `prior_pad()` which pads the state for a single timepoint into that of `n_res` timepoints.  This can be used to facilitate the computation of the other methods.
    """

    def __init__(self, dt, n_res, bootstrap):
        # creates private variable self._bootstrap
        super().__init__(bootstrap=bootstrap)
        # other private variables
        self._dt = dt
        self._n_res = n_res

    def prior_pad(self, x_init):
        """
        Pad the initial state with `n_res - 1` zeros.

        This is helpful for constructing `x_init` to have the correct dimension.

        Parameters
        ----------
        x_init: PyTree
            The state variable at time `t = 0`.

        Returns
        -------
        x_init: PyTree
            The original `x_init` of which each leaf has been zero-padded to have leading dimension `n_res`.
        """

        def fun(x):
            """Prepend zeros along leading dimension."""
            zeros = jnp.zeros((self._n_res - 1,) + x.shape)
            return jnp.concatenate([zeros, x[None]])

        return jax.tree.map(fun, x_init)

    def state_dt_lpdf(self, x_curr, x_prev, dt, theta):
        """
        Log-pdf of state transition over a time interval `dt`.

        Parameters
        ----------
        dt: float
            Time interval between `x_prev` and `x_curr`, which is `self.dt / self.n_res`.
        """
        raise NotImplementedError

    def state_dt_sample(self, key, x_prev, dt, theta):
        """
        Sample a state transition over a time interval `dt`.
        """
        raise NotImplementedError

    def step_dt_lpdf(self, x_curr, x_prev, y_next, dt_prev, dt_next, theta):
        """
        Calculate the log-pdf of a proposal within a time interval.

        Parameters
        ----------
        y_next: PyTree
            The next measurement, which is at time `dt_prev + dt_next`.
        dt_prev: float
            The time between `x_prev` and `x_curr`.
        dt_next: float
            The time between `x_curr` and `y_next`.
        """
        raise NotImplementedError

    def step_dt_sample(self, key, x_prev, y_next, dt_prev, dt_next, theta):
        """
        Sample a proposal within a time interval.
        """
        raise NotImplementedError

    def pf_step_dt(key, x_prev, y_next, dt_prev, dt_next, theta):
        """
        Particle filter update within a time interval.

        Warnings
        --------
        The returned `logw` **must not include** the contribution from `meas_lpdf()`.  This is because `pf_step_dt()` will be run through `lax.scan()` inside `pf_step()`, and it's simply easier to add the contribution from `meas_lpdf()` after this is done.
        """
        x_curr = self.step_dt_sample(
            key=key,
            x_prev=x_prev,
            y_next=y_next,
            dt_prev=dt_prev,
            dt_next=dt_next,
            theta=theta,
        )
        lp_prop = self.step_dt_lpdf(
            x_curr=x_curr,
            x_prev=x_prev,
            y_next=y_next,
            dt_prev=dt_prev,
            dt_next=dt_next,
            theta=theta,
        )
        lp_targ = self.state_dt_lpdf(
            x_curr=x_curr, x_prev=x_prev, dt=dt_prev, theta=theta
        )
        logw = lp_targ - lp_prop
        return x_curr, logw

    def state_lpdf(self, x_curr, x_prev, theta):
        dt_res = self._dt / self._n_res
        x0 = jax.tree.map(
            lambda xp, xc: jnp.concatenate([xp[-1][None], xc[:-1]]),
            x_prev,
            x_curr,
        )
        x1 = x_curr
        lp = jax.vmap(
            fun=lambda xp, xc: self.state_dt_lpdf(
                x_curr=xc, x_prev=xp, dt=dt_res, theta=theta
            )
        )(x0, x1)
        return jnp.sum(lp)

    def state_sample(self, key, x_prev, theta):
        dt_res = self._dt / self._n_res

        # lax.scan function
        def fun(carry, t):
            key, subkey = jax.random.split(carry["key"])
            x = self.state_dt_sample(
                key=subkey, x_prev=carry["x"], dt=dt_res, theta=theta
            )
            res = {"x": x, "key": key}
            return res, x

        # lax.scan initial value
        init = {
            "x": jax.tree.map(lambda xp: xp[-1], x_prev),
            "key": key,
        }

        # lax.scan itself
        _, full = jax.lax.scan(fun, init, jnp.arange(self._n_res))
        return full

    def step_lpdf(self, x_curr, x_prev, y_curr, theta):
        if self._bootstrap:
            return super().step_lpdf(
                x_curr=x_curr, x_prev=x_prev, y_curr=y_curr, theta=theta
            )
        else:
            x0 = jax.tree.map(
                lambda xp, xc: jnp.concatenate([xp[-1][None], xc[:-1]]),
                x_curr,
                x_prev,
            )
            x1 = x_curr
            dt_res = self._dt / self._n_res
            lp = jax.vmap(
                fun=lambda xp, xc, n: self.step_dt_lpdf(
                    x_curr=xc,
                    x_prev=xp,
                    y_next=y_curr,
                    dt_prev=dt_res,
                    dt_next=self._dt - (n + 1.0) * dt_res,
                    theta=theta,
                )
            )(x0, x1, jnp.arange(self._n_res))
            return jnp.sum(lp)

    def step_sample(self, key, x_prev, y_curr, theta):
        if self._bootstrap:
            return super().step_sample(
                key=key, x_prev=x_prev, y_curr=y_curr, theta=theta
            )
        else:
            dt_res = self._dt / self._n_res

            # lax.scan function
            def fun(carry, n):
                key, subkey = jax.random.split(carry["key"])
                x = self.step_dt_sample(
                    key=subkey,
                    x_prev=carry["x"],
                    y_next=y_curr,
                    dt_prev=dt_res,
                    dt_next=self._dt - (n + 1.0) * dt_res,
                    theta=theta,
                )
                res = {"x": x, "key": key}
                return res, x

            # lax.scan initial value
            init = {
                "x": jax.tree.map(lambda xp: xp[-1], x_prev),
                "key": key,
            }

            # lax.scan itself
            _, full = jax.lax.scan(fun, init, jnp.arange(self._n_res))
            return full

    def pf_step(self, key, x_prev, y_curr, theta):
        if self._bootstrap:
            return super().pf_step(key=key, x_prev=x_prev, y_curr=y_curr, theta=theta)
        else:
            dt_res = self._dt / self._n_res

            # lax.scan function
            def fun(carry, n):
                key = carry["key"]
                x = carry["x"]
                key, subkey = jax.random.split(key)
                x_prop, logw = self.pf_step_dt(
                    key=subkey,
                    x_prev=carry["x"],
                    y_next=y_curr,
                    dt_prev=dt_res,
                    dt_next=self._dt - (n + 1.0) * dt_res,
                    theta=theta,
                )
                res_carry = {"x": x_prop, "key": key}
                res_stack = {"x": x_prop, "logw": logw}
                return res_carry, res_stack

            # lax.scan: initial value
            init = {
                "x": jax.tree.map(lambda xp: xp[self._n_res - 1], x_prev),
                "key": key,
            }

            # lax.scan itself
            _, full = jax.lax.scan(fun, init, jnp.arange(self._n_res))

            # add the contribution of meas_lpdf()
            x_curr = full["x"]
            logw = full["logw"]
            logw = logw + self.meas_lpdf(y_curr=y_curr, x_curr=x_curr, theta=theta)
            return x_curr, logw
