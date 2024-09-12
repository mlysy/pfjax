import jax
import jax.numpy as jnp
import jax.random
from jax.scipy.stats import multivariate_normal


def _additive_gaussian_meas_lpdf(self, y_curr, x_curr, theta):
    """
    Measurement logpdf for additive Gaussian measurements.

    Model is
    ```
    y_curr ~ Normal(wgt * x_curr + offset, var),
    ```
    where `wgt`, `offset`, and `var` are computed by `additive_gaussian_mv()`.
    """
    wgt, offset, var = self.additive_gaussian_mv(
        x_curr=x_curr,
        theta=theta
    )
    mu = jnp.dot(wgt, x_curr) + offset
    return multivariate_normal.logpdf(
        x=y_curr,
        mean=mu,
        cov=var
    )


def _additive_gaussian_meas_sample(self, key, x_curr, theta):
    """
    Measurement logpdf for additive Gaussian measurements.

    Model is
    ```
    y_curr ~ Normal(wgt * x_curr + offset, var),
    ```
    where `wgt`, `offset`, and `var` are computed by `additive_gaussian_mv()`.
    """
    wgt, offset, var = self.additive_gaussian_mv(
        x_curr=x_curr,
        theta=theta
    )
    mu = jnp.dot(wgt, x_curr) + offset
    return jax.random.multivariate_normal(
        key=key,
        mean=mu,
        cov=var
    )


def _fixed_prior_lpdf(self, x_init, theta):
    """
    Compute logpdf for fixed prior.

    TODO: check whether `x_init` is consistent with value given by `fixed_prior_value()`.  This needs to be done carefully to avoid false inequality due to numerical roundoff.
    """
    return jnp.array(0.)


def _fixed_prior_sample(self, key, theta):
    """
    Sample from fixed prior.
    """
    x_init = self.fixed_prior_value(theta=theta)
    return x_init


class BaseModel(object):
    r"""
    Base model for particle filters.

    This class sets up a PF model from small set of methods:

    - The derived class should provide methods `state_lpdf()`, `meas_lpdf()`, `state_sample()` and `meas_sample()` in order calculate the complete data likelihood `pfjax.loglik_full()` and simulate data from the model via `pfjax.simulate()`.

    - To use the "basic" particle filter `pfjax.particle_filter()`, the derived class must provide methods `pf_init()` and `pf_step()`.

    - To use the Rao-Blackwellized particle filter `pfjax.particle_filter_rb()`, the derived class must provide methods `pf_init()`, `step_sample()`, and `step_lpdf()`.

    - If `pf_init()` is missing, the base class will automatically construct it from `prior_lpdf()`, `init_sample()`, and `init_lpdf()`.

    - if `pf_step()` is missing, the base class will automatically construct it from `state_lpdf()`, `meas_lpdf()`, `step_sample()`, and `step_lpdf()`.

    - If in either of the above `step_sample()` and `step_lpdf()` are missing, the base class assumes a bootstrap particle filter and sets `step_sample = state_sample` and `step_lpdf = state_lpdf`.

    - If in either of the above `init_sample()` and `init_lpdf()` are missing, the base calss sets `init_sample = prior_sample` and `init_lpdf = prior_lpdf`.

    - If `additive_gaussian_meas == True`, will create `meas_{sample/lpdf}()` from `additive_gaussian_mv()` method.

    - If `fixed_prior == True`, will create `prior_{lpdf/sample}()` from `fixed_prior_value()` method.

    **Notes: **

    - `pf_step()` and `pf_init()` could be computed more efficiently for bootstrap sampling.  Perhaps this could be specified with an argument `bootstrap` to the constructor.

        Edit: Now done.

    - In general, nothing is stopping the user from creating e.g., `pf_step()` which is inconsistent with `step_sample()`, etc.

    - `additive_gaussian_meas` and `fixed_prior` create methods at runtime, so that no methods are created if these are set to `False`.  However, in order for the user to be able to "overwrite" these at definition time, the methods are created conditionally.

    Args:
        bootstrap: Boolean for whether or not to create a bootstrap particle filter.
        additive_gaussian_meas: Boolean for whether the measurement model is additive Gaussian noise.
        fixed_prior: Boolean for whether the initial state variable is fixed.

    """

    def _cond_setattr(self, name, value):
        """
        Set attribute only if it doesn't already exist.
        """
        if getattr(self.__class__, name, None) is None:
            setattr(self.__class__, name, value)

    def __init__(self, bootstrap, additive_gaussian_meas, fixed_prior):
        self._bootstrap = bootstrap
        self._additive_gaussian_meas = additive_gaussian_meas
        self._fixed_prior = fixed_prior

        # Run-time creation of methods.
        # However, want to avoid creating these if the Derived class
        # has defined them.
        if self._additive_gaussian_meas:
            self._cond_setattr(
                name="meas_lpdf",
                value=_additive_gaussian_meas_lpdf
            )
            self._cond_setattr(
                name="meas_sample",
                value=_additive_gaussian_meas_sample
            )

        if self._fixed_prior:
            self._cond_setattr(
                name="prior_lpdf",
                value=_fixed_prior_lpdf
            )
            self._cond_setattr(
                name="prior_sample",
                value=_fixed_prior_sample
            )

    def step_sample(self, key, x_prev, y_curr, theta):
        r"""
        Sample from default proposal distribution

        : :

            q(x_curr | x_prev, y_curr, theta) = p(x_curr | x_prev, theta)

        Args:
            key: PRNG key.
            x_prev: State variable at previous time `t-1`.
            y_curr: Measurement variable at current time `t`.
            theta: Parameter value.

        Returns:
            Sample of the state variable `x_curr` at current time `t`.
        """
        if self._bootstrap:
            return self.state_sample(key=key, x_prev=x_prev, theta=theta)
        else:
            pass

    def step_lpdf(self, x_curr, x_prev, y_curr, theta):
        r"""
        Calculate log-density of the default proposal distribution

        : :

            q(x_curr | x_prev, y_curr, theta) = p(x_curr | x_prev, theta)

        Args:
            x_curr: State variable at current time `t`.
            x_prev: State variable at previous time `t-1`.
            y_curr: Measurement variable at current time `t`.
            theta: Parameter value.

        Returns:
            Log-density of the state variable `x_curr` at current time `t`.
        """
        if self._bootstrap:
            return self.state_lpdf(x_curr=x_curr, x_prev=x_prev, theta=theta)
        else:
            pass

    def init_sample(self, key, y_init, theta):
        r"""
        Sample from default initial proposal distribution

        : :

            q(x_init | y_init, theta) = p(x_init | theta)

        Args:
            key: PRNG key.
            y_init: Measurement variable at initial time `t = 0`.
            theta: Parameter value.

        Returns:
            Sample of the state variable `x_init` at initial time `t = 0`.
        """
        if self._bootstrap:
            return self.prior_sample(key=key, theta=theta)
        else:
            pass

    def init_lpdf(self, x_init, y_init, theta):
        r"""
        Calculate log-density of the default proposal distribution

        : :

            q(x_curr | x_prev, y_curr, theta) = p(x_curr | x_prev, theta)

        Args:
            x_init: State variable at initial time `t = 0`.
            y_init: Measurement variable at initial time `t = 0`.
            theta: Parameter value.

        Returns:
            Log-density of the state variable `x_init` at initial time `t = 0`.
        """
        if self._bootstrap:
            return self.prior_lpdf(x_init=x_init, theta=theta)
        else:
            pass

    def pf_step(self, key, x_prev, y_curr, theta):
        r"""
        Particle filter update.

        Returns a draw from proposal distribution

        : :

            x_curr ~ q(x_curr) = q(x_curr | x_prev, y_curr, theta)

        and the log weight

        : :

            logw = log p(y_curr | x_curr, theta) + log p(x_curr | x_prev, theta) - log q(x_curr)

        Args:
            key: PRNG key.
            x_prev: State variable at previous time `t-1`.
            y_curr: Measurement variable at current time `t`.
            theta: Parameter value.

        Returns:
            Tuple:

            - **x_curr ** - A sample from the proposal distribution at current time `t`.
            - **logw ** - The log-weight of `x_curr`.
        """

        if self._bootstrap:
            x_curr = self.state_sample(key=key, x_prev=x_prev, theta=theta)
            logw = self.meas_lpdf(y_curr=y_curr, x_curr=x_curr, theta=theta)
        else:
            x_curr = self.step_sample(key=key, x_prev=x_prev,
                                      y_curr=y_curr, theta=theta)
            lp_prop = self.step_lpdf(x_curr=x_curr,
                                     x_prev=x_prev, y_curr=y_curr, theta=theta)
            lp_targ = self.state_lpdf(
                x_curr=x_curr, x_prev=x_prev, theta=theta
            ) + self.meas_lpdf(y_curr=y_curr, x_curr=x_curr, theta=theta)
            logw = lp_targ - lp_prop
        return x_curr, logw

    def pf_init(self, key, y_init, theta):
        r"""
        Initial step of particle filter.

        Returns a draw from the proposal distribution

        : :

            x_init ~ q(x_init) = q(x_init | y_init, theta)

        and calculates the log weight

        : :

            logw = log p(y_init | x_init, theta) + log p(x_init | theta) - log q(x_init)

        Args:
            key: PRNG key.
            y_init: Measurement variable at initial time `t = 0`.
            theta: Parameter value.

        Returns:
            Tuple:

            - **x_init ** - A sample from the proposal distribution at initial tme `t = 0`.
            - **logw ** - The log-weight of `x_init`.
        """
        if self._bootstrap:
            x_curr = self.prior_sample(key=key, theta=theta)
            logw = self.meas_lpdf(y_curr=y_init, x_curr=x_curr, theta=theta)
        else:
            x_curr = self.init_sample(key=key, theta=theta)
            lp_prop = self.init_lpdf(x_curr=x_curr, theta=theta)
            lp_targ = self.prior_lpdf(x_curr=x_curr, theta=theta) +
            self.meas_lpdf(y_curr=y_init, x_curr=x_curr, theta=theta)
            logw = lp_targ - lp_prop
        return x_curr, logw
