import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
# from .base_model import BaseModel
import pfjax as pf

class SSModel(pf.BaseModel):
    def __init__(self, n_state, n_meas):
        """
        Model:
        x_t = A x_t-1 + b + C * eps_t
        y_t = D x_t + e + F * eta_t

        For this model, theta needs to be passed as a tuple: theta = tuple(A, b, C, D, e, F).
        Args:
            n_state: Dimension of the state variable.
            n_meas: Dimension of the measurement variable.
        """
        super().__init__(bootstrap=True)
        # prepend underscore to internal version so users can't modify 
        # directly without considerable effort.
        # (things will fail down the road if they do)
        self._n_state = n_state
        self._n_meas = n_meas 

    def unpack_theta(self, theta):
        A, b, C, D, e, F = theta
        # check that shape of A is correct (don't do this in real code)
        # assert A.shape == (self._n_state, self._n_state)
        return A, b, C, D, e, F

    def prior_lpdf(self, x_init, theta):
        # Needed for initial state x_0
        A, b, C, D, e, F = self.unpack_theta(theta)
        mean_x = b
        cov_x = C @ C.T
        return jsp.stats.multivariate_normal.logpdf(x_init, mean_x, cov_x)

    def prior_sample(self, key, theta):
        A, b, C, D, e, F = self.unpack_theta(theta)
        mean_x = b
        noise_x = jax.random.normal(key, shape=(self._n_state,))
        return mean_x + C.T @ noise_x

    def state_lpdf(self, x_curr, x_prev, theta):
        # Needed for x_t | x_{t-1}
        A, b, C, D, e, F = self.unpack_theta(theta)
        mean_x = A @ x_prev + b
        cov_x = C @ C.T
        return jsp.stats.multivariate_normal.logpdf(x_curr, mean_x, cov_x)

    def state_sample(self, key, x_prev, theta):
        # Sample from x_t | x_{t-1}
        A, b, C, D, e, F = self.unpack_theta(theta)
        mean_x = A @ x_prev + b
        noise_x = jax.random.normal(key, shape=(self._n_state,))
        return mean_x + C.T @ noise_x

    def meas_lpdf(self, y_curr, x_curr, theta):
        # Needed for y_t | x_t
        A, b, C, D, e, F = self.unpack_theta(theta)
        mean_y = D @ x_curr + e
        cov_y = F @ F.T
        return jsp.stats.multivariate_normal.logpdf(y_curr, mean_y, cov_y)

    def meas_sample(self, key, x_curr, theta):
        # Sample from y_t | x_t
        A, b, C, D, e, F = self.unpack_theta(theta)
        mean_y = D @ x_curr + e
        noise_y = jax.random.normal(key, shape=(self._n_meas,))
        return mean_y + F.T @ noise_y
