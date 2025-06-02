class BaseModel(object):
    r"""
    Base class for defining state-space models used in particle filters.

    This class constructs a PF model from a minimal set of methods.

    Required methods by use case
    ----------------------------
    - To compute the complete data likelihood with `pfjax.loglik_full()` and to simulate data
      with `pfjax.simulate()`, the derived class must implement:

        - state_lpdf()
        - meas_lpdf()
        - state_sample()
        - meas_sample()

    - To use the basic particle filter via `pfjax.particle_filter()`, the derived class must implement:

        - pf_init()
        - pf_step()

    - To use the Rao-Blackwellized particle filter via `pfjax.particle_filter_rb()`, the derived class must implement:

        - pf_init()
        - step_sample()
        - step_lpdf()

    Optional behavior
    -----------------
    If the following methods are not provided by the subclass, they will be automatically constructed:

    - If `pf_init()` is missing:
        - Built from: `prior_lpdf()`, `init_sample()`, and `init_lpdf()`

    - If `pf_step()` is missing:
        - Built from: `state_lpdf()`, `meas_lpdf()`, `step_sample()`, and `step_lpdf()`

    - If `step_sample()` or `step_lpdf()` are missing:
        - Assumes a bootstrap filter:
            - `step_sample = state_sample`
            - `step_lpdf = state_lpdf`

    - If `init_sample()` or `init_lpdf()` are missing:
        - Uses:
            - `init_sample = prior_sample`
            - `init_lpdf = prior_lpdf`

    Notes
    -----
    - For bootstrap filters, `pf_step()` and `pf_init()` can be computed more efficiently. 
      This can be specified with an argument `bootstrap` to the constructor.
    
    - Users are responsible for ensuring consistency between manually defined methods. 
      (e.g., consistency between `pf_step()` and `step_sample()`, etc.)

    Parameters
    ----------
    bootstrap : bool
        Whether or not to create a bootstrap particle filter.

    """

    def __init__(self, bootstrap):
        self._bootstrap = bootstrap

    def step_sample(self, key, x_prev, y_curr, theta):
        r"""
        Sample from the default proposal distribution:
        
        ::

            q(x_curr | x_prev, y_curr, theta) = p(x_curr | x_prev, theta)
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key.
        x_prev : array-like
            State variable at previous time t - 1.
        y_curr : array-like
            Measurement variable at current time t.
        theta : array-like
            Parameter vector.

        Returns
        -------
        x_curr : ndarray
            Sample of the state variable ``x_curr`` at current time t.
        
        """
        if self._bootstrap:
            return self.state_sample(key=key, x_prev=x_prev, theta=theta)
        else:
            pass

    def step_lpdf(self, x_curr, x_prev, y_curr, theta):
        r"""
        Calculate log-density of the default proposal distribution.

        ::

            q(x_curr | x_prev, y_curr, theta) = p(x_curr | x_prev, theta)

        Parameters
        ----------
        x_curr : array-like
            State variable at current time t.
        x_prev : array-like
            State variable at previous time t - 1.
        y_curr : array-like
            Measurement variable at current time t.
        theta : array-like
            Parameter value.

        Returns
        -------
        float
            Log-density of the state variable ``x_curr`` at current time t.

        """
        if self._bootstrap:
            return self.state_lpdf(x_curr=x_curr, x_prev=x_prev, theta=theta)
        else:
            pass

    def init_sample(self, key, y_init, theta):
        r"""
        Sample from default initial proposal distribution.

        ::

            q(x_init | y_init, theta) = p(x_init | theta)

        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key.
        y_init : array-like
            Measurement variable at initial time t = 0.
        theta : array-like
            Parameter value.

        Returns
        -------
        x_init : ndarray
            Sample of the state variable ``x_init`` at initial time t = 0.
        
        """
        if self._bootstrap:
            return self.prior_sample(key=key, theta=theta)
        else:
            pass

    def init_lpdf(self, x_init, y_init, theta):
        r"""
        Calculate log-density of the default proposal distribution.

        ::

            q(x_curr | x_prev, y_curr, theta) = p(x_curr | x_prev, theta)

        Parameters
        ----------
        x_init : array-like
            Latent state variable at initial time t = 0.
        y_init : array-like
            Measurement variable at initial time t = 0.
        theta : array-like
            Parameter value.

        Returns
        -------
        float
            Log-density of the state variable ``x_init`` at initial time t = 0.

        """
        if self._bootstrap:
            return self.prior_lpdf(x_init=x_init, theta=theta)
        else:
            pass

    def pf_step(self, key, x_prev, y_curr, theta):
        r"""
        Performs a particle filter update.

        Returns a draw from proposal distribution

        ::

            x_curr ~ q(x_curr | x_prev, y_curr, theta)

        and the log-weight
        
        ::

            logw = log p(y_curr | x_curr, theta) + log p(x_curr | x_prev, theta) - log q(x_curr)

        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key.
        x_prev : array-like
            State variable at previous time t - 1.
        y_curr : array-like
            Measurement variable at current time t.
        theta : array-like
            Parameter value.

        Returns
        -------
        x_curr : ndarray
            A sample from the proposal distribution at current time t.
        logw : float
            The log-weight of ``x_curr``.

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
        Performs the initial step of the particle filter.

        Returns a draw from the proposal distribution

        ::

            x_init ~ q(x_init) = q(x_init | y_init, theta)

        and calculates the log weight

        ::

            logw = log p(y_init | x_init, theta) + log p(x_init | theta) - log q(x_init)

        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key.
        y_init : array-like
            Measurement at initial time t = 0.
        theta : array-like
            Parameter value.

        Returns
        -------
        x_init : ndarray
            A Sample from the initial proposal distribution at time t = 0.
        logw : float
            The log-weight of ``x_init``.

        """
        if self._bootstrap:
            x_curr = self.prior_sample(key=key, theta=theta)
            logw = self.meas_lpdf(y_curr=y_init, x_curr=x_curr, theta=theta)
        else:
            x_curr = self.init_sample(key=key, theta=theta)
            lp_prop = self.init_lpdf(x_curr=x_curr, theta=theta)
            lp_targ = self.prior_lpdf(x_curr=x_curr, theta=theta) + \
                self.meas_lpdf(y_curr=y_init, x_curr=x_curr, theta=theta)
            logw = lp_targ - lp_prop
        return x_curr, logw
