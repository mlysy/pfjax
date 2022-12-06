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

    **Notes:**

    - `pf_step()` and `pf_init()` could be computed more efficiently for bootstrap sampling.  Perhaps this could be specified with an argument `bootstrap` to the constructor.  

    - In general, nothing is stopping the user from creating e.g., `pf_step()` which is inconsistent with `step_sample()`, etc.

    Args:
        bootstrap: Boolean for whether or not to create a bootstrap particle filter.

    """

    def __init__(self, bootstrap):
        self._bootstrap = bootstrap

    def step_sample(self, key, x_prev, y_curr, theta):
        r"""
        Sample from default proposal distribution

        ::

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

        ::

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

        ::

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

        ::

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

        ::

            x_curr ~ q(x_curr) = q(x_curr | x_prev, y_curr, theta)

        and the log weight

        ::

            logw = log p(y_curr | x_curr, theta) + log p(x_curr | x_prev, theta) - log q(x_curr)

        Args:
            key: PRNG key.
            x_prev: State variable at previous time `t-1`.
            y_curr: Measurement variable at current time `t`.
            theta: Parameter value.

        Returns:
            Tuple:

            - **x_curr** - A sample from the proposal distribution at current time `t`.
            - **logw** - The log-weight of `x_curr`.
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

        ::

            x_init ~ q(x_init) = q(x_init | y_init, theta)

        and calculates the log weight

        ::

            logw = log p(y_init | x_init, theta) + log p(x_init | theta) - log q(x_init)

        Args:
            key: PRNG key.
            y_init: Measurement variable at initial time `t = 0`.
            theta: Parameter value.

        Returns:
            Tuple:

            - **x_init** - A sample from the proposal distribution at initial tme `t = 0`.
            - **logw** - The log-weight of `x_init`.
        """
        if self._bootstrap:
            x_init = self.prior_sample(key=key, theta=theta)
            logw = self.meas_lpdf(y_curr=y_init, x_curr=x_init, theta=theta)
        else:
            x_init = self.init_sample(key=key, y_init=y_init, theta=theta)
            lp_prop = self.init_lpdf(x_init=x_init, y_init=y_init, theta=theta)
            lp_targ = self.prior_lpdf(x_init=x_init, theta=theta) + \
                self.meas_lpdf(y_curr=y_init, x_curr=x_init, theta=theta)
            logw = lp_targ - lp_prop
        return x_init, logw
