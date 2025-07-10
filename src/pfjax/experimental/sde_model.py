import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pfjax.experimental.continuous_time_model
import pfjax.mvn_bridge


class SDEModel(pfjax.experimental.continuous_time_model.ContinuousTimeModel):
    r"""
    Base class for SDE models.

    This class inheriting from `ContinuousTimeModel` sets up `state_dt_{lpdf/sample}()` automatically from user-specified SDE drift and diffusion functions via the Euler approximation.

    For computational efficiency, the user can also specify whether or not the diffusion is diagonal.

    The class inherits a `bootstrap` argument from `ContinuousTimeModel`.  When `bootstrap=False`, `pf_step_dt()` implements the bridge proposal of Durham-Gallant (2002).  This requires the method `bridge_pars()` to be defined.

    TODO:

    - [ ] Implement the state validator `is_valid()`.  This should probably be applied after every call to `state_dt_sample()`, `step_dt_sample()`, and `pf_step_dt()`.  However, dealing with PyTrees, nans, etc is a bit annoying right now.

    - [x] Figure out how to do `bootstrap` argument correctly.  In a nutshell, using the bootstrap is decided after the specific model is created, rather than the other way around.

        Current solution defines `step_dt_{lpdf/sample}()` from `bridge_pars()`.  Should then use `bootstrap` argument inherited from `ContinuousTimeModel` class...
    """

    def __init__(self, dt, n_res, diff_diag, bootstrap):
        r"""
        Class constructor.

        Parameters
        ----------
        dt : float
            SDE interobservation time.
        n_res : int
            SDE resolution number.  There are `n_res` latent variables per observation, equally spaced with interobservation time `dt/n_res`.
        diff_diag : bool
            Whether or not the diffusion matrix is assumed to be diagonal.
        bootstrap : bool
            Whether or not to use a bootstrap filter.  If `bootstrap == False`, use a bridge proposal, for which we must define the `bridge_pars()` method.
        """
        self._diff_diag = diff_diag
        super().__init__(dt=dt, n_res=n_res, bootstrap=bootstrap)

    def _state_dt_pars(self, x_prev, dt, theta):
        """Compute drift and diffusion parameters."""
        dr = x_prev + self.drift(x=x_prev, theta=theta) * dt
        df = self.diff(x=x_prev, theta=theta)
        if self._diff_diag:
            df = df * jnp.sqrt(dt)
        else:
            df = df * dt
        return dr, df

    def state_dt_lpdf(self, x_curr, x_prev, dt, theta):
        dr, df = self._state_dt_pars(x_prev=x_prev, dt=dt, theta=theta)
        if self._diff_diag:
            return jnp.sum(jsp.stats.norm.logpdf(x=x_curr, loc=dr, scale=df))
        else:
            return jsp.stats.multivariate_normal.logpdf(x=x_curr, mean=dr, cov=df)

    def state_dt_sample(self, key, x_prev, dt, theta):
        dr, df = self._state_dt_pars(x_prev=x_prev, dt=dt, theta=theta)
        if self._diff_diag:
            return dr + df * jax.random.normal(key, (dr.shape[0],))
        else:
            return jax.random.multivariate_normal(key, mean=dr, cov=df)

    def bridge_pars(self, y_curr, theta):
        """
        Compute the parameters of the bridge proposal.

        This proposal assumes that `meas_lpdf(y_curr, x_curr, theta)` as a function of `x_curr` by

        ```
        Y ~ Normal(A x_curr[-1], Omega),
        ```

        where `Y`, `A`, and `Omega` are functions of `y_curr` and `theta`.

        Returns
        -------
        Y : Array
            Pseudo-observation.
        A : Array
            Pseudo-weight matrix.
        Omega: Array
            Pseudo-variance matrix.
        """
        raise NotImplementedError

    def _step_dt_pars(self, x_prev, y_next, dt_prev, dt_next, theta):
        """Parameters of the normal distribution for step_dt."""
        # pseudo linear gaussian measurement model
        Y, A, Omega = self.bridge_pars(y_curr=y_next, theta=theta)
        # drift and diffusion (variance scale)
        dr = self.drift(x=x_prev, theta=theta)
        df = self.diff(x=x_prev, theta=theta)
        if self._diff_diag:
            df = jnp.diag(df)
        df = df * dt_prev
        return pfjax.mvn_bridge.mvn_bridge_mv(
            mu_W=x_prev + dr * dt_prev,
            Sigma_W=df,
            mu_Y=jnp.matmul(A, x_prev + dr * dt_next),
            AS_W=jnp.matmul(A, df),
            Sigma_Y=k * jnp.linalg.multi_dot([A, df, A.T]) + Omega,
            Y=Y,
        )

    def step_dt_lpdf(self, x_curr, x_prev, y_next, dt_prev, dt_next, theta):
        mu, Sigma = self._step_dt_pars(
            x_prev=x_prev,
            y_next=y_next,
            dt_prev=dt_prev,
            dt_next=dt_next,
            theta=theta,
        )
        return jsp.stats.multivariate_normal.logpdf(x=x_curr, mean=mu, cov=Sigma)

    def step_dt_sample(self, key, x_prev, y_next, dt_prev, dt_next, theta):
        mu, Sigma = self._step_dt_pars(
            x_prev=x_prev,
            y_next=y_next,
            dt_prev=dt_prev,
            dt_next=dt_next,
            theta=theta,
        )
        return jax.random.multivariate_normal(key=key, mean=mu, cov=Sigma)
