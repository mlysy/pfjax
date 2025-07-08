import jax
import jax.numpy as jnp

# --- building classes ---------------------------------------------------------


class foo:
    def bar(self, x):
        pass


baz = foo()
y = baz.bar(3)
y


def foo(x):
    return x + 5.0


class bar:
    baz = foo(3.0)


y = bar()

y.baz(3.0)

# --- conditional method definition --------------------------------------------


def f(self, x):
    return x + 3


def g(self, x):
    return x * 2.5


def h(x):
    return f if x > 5.0 else g


class Foo(object):
    __call__ = h(2)


fun = Foo()
fun(6.7)

# --- test inherited members ---------------------------------------------------


class Base(object):
    def __init__(self, x):
        self.x = x


class Derived(Base):
    def __init__(self, x, y):
        super().__init__(x=x)
        self.y = y


foo = Derived(x=2, y=3)


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


# --- test two-way strategy pattern --------------------------------------------


from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


class Distribution:
    def lpdf(self, x, *args, **kwargs):
        raise NotImplementedError

    def sample(self, key, *args, **kwargs):
        raise NotImplementedError


@dataclass
class NormalDistribution(Distribution):
    pars: Callable

    def lpdf(self, x, *args, **kwargs):
        loc, scale = self.pars(*args, **kwargs)
        return jax.scipy.stats.norm.logpdf(x=x, loc=loc, scale=scale)

    def sample(self, key, *args, **kwargs):
        loc, scale = self.pars(*args, **kwargs)
        return loc + scale * jax.random.normal(key=key)


@dataclass
class BaseModel(object):
    meas: Distribution | None = None

    def meas_lpdf(self, y_curr, x_curr, theta):
        if self.meas is not None:
            return self.meas.lpdf(x=y_curr, x_curr=x_curr, theta=theta)
        else:
            raise NotImplementedError

    def meas_sample(self, key, x_curr, theta):
        if self.meas is not None:
            return self.meas.sample(key=key, x_curr=x_curr, theta=theta)


@dataclass
class BMModel(BaseModel):
    meas = NormalDistribution(pars=self.meas_pars)
    # def __init__(self):
    #     super().__init__(
    #         meas=NormalDistribution(pars=self.meas_pars),
    #     )

    def meas_pars(self, x_curr, theta):
        return x_curr, theta


bm_model = BMModel()

y_curr = 3.0
x_curr = 2.0
theta = 1.0

bm_model.meas_lpdf(y_curr=y_curr, x_curr=x_curr, theta=theta)

jax.scipy.stats.norm.logpdf(x=y_curr, loc=x_curr, scale=theta)
