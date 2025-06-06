import jax
import jax.numpy as jnp
import jax.random


class BaseModel(object):
    r"""
    Base model for particle filters.

    This class sets up a PF model from small set of methods:

    - To simulate data with `pfjax.simulate()`, the derived class must provide methods `state_sample()` and `meas_sample()`, and optionally `prior_sample()`.

    - To evaluate the complete-data logliklihood with `pfjax.loglik_full()`, the derived class must provide methods `prior_lpdf()`, `state_lpdf()` and `meas_lpdf()`.

    - To use the Basic particle filter `pfjax.particle_filter()`, the derived class must provide methods `pf_init()` and `pf_step()`.

    - To use the Rao-Blackwellized particle filter `pfjax.particle_filter_rb()`, the derived class must provide methods `pf_init()`, `step_sample()`, and `step_lpdf()`.

    - If `bootstrap=False`:

        - `pf_init()` is automatically constructed from `prior_lpdf()`, `init_sample()`, and `init_lpdf()`.

        - `pf_step()` is automatically constructed from `state_lpdf()`, `meas_lpdf()`, `step_sample()`, and `step_lpdf()`.

    - If `bootstrap=True`:

        - `pf_init()` is automatically constructed from `prior_sample()` and `meas_lpdf()`.

        - `pf_step()` is automatically constructed from `step_sample()` and `meas_lpdf()`.

        - The class sets `step_sample=state_sample` and `step_lpdf=state_lpdf`.

        - The class sets `init_sample=prior_sample` and `init_lpdf=prior_lpdf`.


    PARAMETERS
    ----------
    bootstrap: bool
        Whether or not to create a bootstrap particle filter.
    """

    def __init__(self, bootstrap):
        self._bootstrap = bootstrap

    def step_sample(self, key, x_prev, y_curr, theta):
        if self._bootstrap:
            return self.state_sample(key=key, x_prev=x_prev, theta=theta)
        else:
            raise NotImplementedError

    def step_lpdf(self, x_curr, x_prev, y_curr, theta):
        if self._bootstrap:
            return self.state_lpdf(x_curr=x_curr, x_prev=x_prev, theta=theta)
        else:
            raise NotImplementedError

    def init_sample(self, key, y_init, theta):
        if self._bootstrap:
            return self.prior_sample(key=key, theta=theta)
        else:
            raise NotImplementedError

    def init_lpdf(self, x_init, y_init, theta):
        if self._bootstrap:
            return self.prior_lpdf(x_init=x_init, theta=theta)
        else:
            raise NotImplementedError

    def pf_step(self, key, x_prev, y_curr, theta):
        if self._bootstrap:
            x_curr = self.state_sample(key=key, x_prev=x_prev, theta=theta)
            logw = self.meas_lpdf(y_curr=y_curr, x_curr=x_curr, theta=theta)
        else:
            x_curr = self.step_sample(
                key=key, x_prev=x_prev, y_curr=y_curr, theta=theta
            )
            lp_prop = self.step_lpdf(
                x_curr=x_curr, x_prev=x_prev, y_curr=y_curr, theta=theta
            )
            lp_targ = self.state_lpdf(x_curr=x_curr, x_prev=x_prev, theta=theta)
            lp_targ = lp_targ + self.meas_lpdf(
                y_curr=y_curr, x_curr=x_curr, theta=theta
            )
            logw = lp_targ - lp_prop
        return x_curr, logw

    def pf_init(self, key, y_init, theta):
        if self._bootstrap:
            x_curr = self.prior_sample(key=key, theta=theta)
            logw = self.meas_lpdf(y_curr=y_init, x_curr=x_curr, theta=theta)
        else:
            x_curr = self.init_sample(key=key, theta=theta)
            lp_prop = self.init_lpdf(x_curr=x_curr, theta=theta)
            lp_targ = self.prior_lpdf(x_curr=x_curr, theta=theta)
            lp_targ = lp_targ + self.meas_lpdf(
                y_curr=y_init, x_curr=x_curr, theta=theta
            )
            logw = lp_targ - lp_prop
        return x_curr, logw
