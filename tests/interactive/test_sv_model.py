import jax
import jax.numpy as jnp


class StoVolModel(sde.SDEModel):
    def __init__(self, n_res, dt, x_init, eps=0.00001):
        """
        Parameters
        ----------
        x_init: jax.Array
            Vector of length two specifying the initial value of the SDE (considered fixed and known).
        """
        super().__init__(dt, n_res, diff_diag=False, bootstrap=False)
        self._eps = eps

    def drift(self, x, theta):
        alpha, gamma, eta, sigma, rho = theta
        return jnp.array([alpha - 0.5 * jnp.exp(2 * x[1]), -gamma * x[1] + eta])

    def diff(self, x, theta):
        alpha, gamma, eta, sigma, rho = theta
        return jnp.array(
            [
                [jnp.exp(2 * x[1]), rho * jnp.exp(x[1]) * sigma],
                [rho * jnp.exp(x[1]) * sigma, sigma**2],
            ]
        )

    def bridge_pars(self, y_curr, theta):
        A = jnp.array([1.0, 0.0])
        Omega = eps**2
        Y = y_curr
        return A, Omega, Y

    def meas_pars(self, x_curr, theta):
        A, Omega, _ = self.bridge_pars(y_curr=None, theta=theta)
        mu = jnp.dot(A, x_curr[-1])
        sigma = jnp.sqrt(Omega)
        return mu, sigma

    def meas_sample(self, key, x_curr, theta):
        # just return the last log-asset value
        # x_curr is an array of shape `(n_res, 2)`.
        mu, sigma = self.meas_pars(x_curr=x_curr, theta=theta)
        return mu + sigma * jax.random.normal(key)

    def meas_lpdf(self, x_curr, y_curr, theta):
        mu, sigma = self.meas_pars(x_curr=x_curr, theta=theta)
        return jsp.stats.norm.logpdf(x=y_curr, loc=mu, scale=sigma)

    def pf_step(self, key, x_prev, y_curr, theta):
        A, Omega, Y = self.bridge_pars(y_curr=y_curr, theta=theta)
        return self.bridge_step(
            key=key, x_prev=x_prev, y_curr=y_curr, theta=theta, Y=Y, A=A, Omega=Omega
        )


# true parameters
alpha = 0.03
gamma = 0.01
eta = 0.02
sigma = 0.04
rho = 0.3
theta_true = jnp.array([alpha, gamma, eta, sigma, rho])

# data specification
dt = 0.5
n_obs = 100
x_init = jnp.array([0.0, 0.0])
key = jax.random.PRNGKey(0)

# simulation
sv_model = StoVolModel(dt=dt)
sv_model.state_sample(key=key, x_prev=x_init, theta=theta_true)

a = pfjax.sde.SDEModel(dt=0.1, n_res=1, diff_diag=True)
b = pfjax.sde.SDEModel(dt=0.1, n_res=1, diff_diag=False)


# --- test whether args/kwargs can be missing altogether -----------------------


def foo(x, bar, *args, **kwargs):
    return bar(x, *args, **kwargs)


def bar(x, y):
    return x + y + 5.0


foo(5.0, bar, 17.0)


# --- test whether we can confuse a base class ---------------------------------

import jax
import jax.numpy as jnp


class Base(object):
    def foo(self, x):
        return jnp.sin(x)

    def bar(self, x):
        return jax.tree.map(self.foo, x)


class Derived(Base):
    def __init__(self, use_base_foo):
        self.use_base_foo = use_base_foo

    def foo(self, x):
        if self.use_base_foo:
            return super().foo(x)
        else:
            return jnp.exp(x)


key = jax.random.PRNGKey(0)

x = (jnp.arange(3), jax.random.normal(key, (2, 5)), {"y": jax.random.normal(key, (3,))})

obj = Derived(use_base_foo=False)
obj.bar(x)


class A:
    def greet(self):
        return "A"


class B(A):
    def __init__(self, use_base_greet):
        self.use_base_greet = use_base_greet

    def greet(self):
        if self.use_base_greet:
            return super().greet()
        else:
            return "B"


foo = B(use_base_greet=True)
foo.greet()

bar = B(use_base_greet=False)
bar.greet()

# --- test replication ---------------------------------------------------------


def zero_pad(x, n):
    """Zero-pad an array along the leading dimension."""
    zeros = jnp.zeros((n - 1,) + x.shape)
    return jnp.concatenate([zeros, x[None]])


jax.jit(zero_pad, static_argnums=1)(x=jnp.ones((3, 2)), n=1)
