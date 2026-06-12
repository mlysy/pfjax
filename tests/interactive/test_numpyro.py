# --- using numpyro to create _sim and _lpdf -----------------------------------

# Works fine for random quantity as a JAX array.
# Not sure how to automate this when its a PyTree.

import inspect

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy
import numpyro
import numpyro.distributions as dist
from jax.scipy.stats import norm as norm_dist
from numpyro import handlers


def state_dist(x_prev, theta, x_curr=None, key=None):
    return numpyro.sample(
        name="x_curr", fn=dist.Normal(loc=x_prev, scale=theta), obs=x_curr, rng_key=key
    )


@jax.jit
def state_sim(key, x_prev, theta):
    return state_dist(key=key, x_prev=x_prev, theta=theta)


key = random.PRNGKey(1)
x_curr = 6.2
x_prev = 5.1
theta = 1.7

state_sim(key=key, x_prev=x_prev, theta=theta)
x_prev + theta * random.normal(key)


def state_lpdf(x_curr, x_prev, theta):
    model = handlers.substitute(
        fn=handlers.seed(state_dist, random.PRNGKey(0)), data={"x_curr": x_curr}
    )
    model_trace = handlers.trace(model).get_trace(
        x_prev=x_prev,
        theta=theta,
    )
    obs_node = model_trace["x_curr"]
    return obs_node["fn"].log_prob(obs_node["value"])


state_lpdf(x_curr=x_curr, x_prev=x_prev, theta=theta)
norm_dist.logpdf(x=x_curr, loc=x_prev, scale=theta)


# ok now let's try with a PyTree


def state_dist(x_prev, theta, x_curr=None, key=None):
    x = numpyro.sample(
        name="x", fn=dist.Normal(loc=x_prev, scale=theta), obs=x_curr, rng_key=key
    )
    y = numpyro.sample(
        name="y", fn=dist.Normal(loc=x, scale=x_prev**2), rng_key=key  # WATCH OUT...
    )
    return x, y


x_prev = 5.1
theta = 1.7
x = 1.0
y = 2.0
model = handlers.substitute(
    fn=handlers.seed(state_dist, random.PRNGKey(0)),
    data={"x": x, "y": y},
)
model_trace = handlers.trace(model).get_trace(
    x_prev=x_prev,
    theta=theta,
)
ll = 0.0
for k in model_trace.keys():
    obs_node = model_trace[k]
    ll = ll + obs_node["fn"].log_prob(obs_node["value"])

ll
norm_dist.logpdf(x, loc=x_prev, scale=theta) + norm_dist.logpdf(
    y, loc=x, scale=x_prev**2
)

# ok let's generalize this


def pyro_lpdf(data, dist, *args, **kwargs):
    """
    Parameters
    ----------
    data: dict
        Argument to the distribution.
    dist: Callable
        Pyro distribution.
    """
    model = handlers.substitute(
        fn=handlers.seed(dist, rng_seed=0),
        data=data,
    )
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    ## don't do it this way because model_trace.keys()
    ## is not an Array
    # def loglik(k):
    #     obs_node = model_trace[k]
    #     return obs_node["fn"].log_prob(obs_node["value"])
    # ll = jax.vmap(loglik)(model_trace.keys())
    # return jnp.sum(ll)
    ll = 0.0
    for k in model_trace.keys():
        obs_node = model_trace[k]
        ll = ll + obs_node["fn"].log_prob(obs_node["value"])
    return ll


def pyro_sample(key, dist, *args, **kwargs):
    model = handlers.seed(dist, rng_seed=key)
    return model(*args, **kwargs)


# --- example: data is a dict -----------------------------------------------


def pyro_dist(x_prev, theta, x_curr={"x": None, "y": None}):
    x = numpyro.sample(
        name="x",
        fn=dist.Normal(loc=x_prev, scale=theta),
        obs=x_curr["x"],
    )
    y = numpyro.sample(
        name="y", fn=dist.Normal(loc=x, scale=x_prev**2), obs=x_curr["y"]
    )
    return {"x": x, "y": y}


key = jax.random.PRNGKey(0)
x_prev = 5.1
theta = 1.7
x_curr = {"x": 1.0, "y": 2.0}

jax.jit(pyro_lpdf, static_argnums=1)(
    data=x_curr, dist=pyro_dist, x_prev=x_prev, theta=theta
)

jax.jit(pyro_sample, static_argnums=1)(
    key=key, dist=pyro_dist, x_prev=x_prev, theta=theta
)

# --- example: x_state and y_meas are PyTrees ----------------------------------


def resolve_kwargs(fun_sig, args, kwargs):
    """Resolve input arguments to positional following a function's signature.

    This will raise a TypeError if any keyword-only arguments were passed by the
    caller.

    Mostly copied from
    <https://github.com/jax-ml/jax/blob/50476ee1fa9009bfb8c63567681af72b1ada5ef4/jax/_src/api_util.py>
    """
    # if isinstance(fun, partial):
    #     # functools.partial should have an opaque signature.
    #     fun = lambda *args, **kwargs: None
    ba = fun_sig.bind(*args, **kwargs)
    ba.apply_defaults()
    if ba.kwargs:
        passed_kwargs = [k for k in ba.kwargs if k in kwargs]
        if passed_kwargs:
            raise TypeError(
                "The following keyword arguments could not be resolved to positions: "
                f"{', '.join(passed_kwargs)}"
            )
    return ba.args


def meas_dist(x_curr, theta, y_curr=(None, None)):
    x1, x2 = x_curr
    t1, t2 = theta
    y1 = numpyro.sample(
        name="y1",
        fn=dist.Normal(loc=x1, scale=t1),
        obs=y_curr[0],
    )
    y2 = numpyro.sample(
        name="y2",
        fn=dist.Normal(loc=x2 * y1, scale=t2 * y1**2),
        obs=y_curr[1],
    )
    return (y1, y2)


def make_dist_sample(dist):
    # create the right signature
    sig = inspect.signature(dist)
    params = list(sig.parameters.values())
    key_param = inspect.Parameter(
        name="key",
        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    new_params = [key_param] + params[:-1]
    new_sig = sig.replace(parameters=params)

    def _sample(key, *args, **kwargs):
        return pyro_sample(key=key, dist=dist, *args, **kwargs)

    _sample.__signature__ = new_sig

    return _sample


def make_dist_lpdf(dist):
    # create signature
    sig = inspect.signature(dist)
    params = list(sig.parameters.values())
    obs_param = params[-1].replace(default=inspect.Parameter.empty)
    new_params = [obs_param] + params[:-1]
    new_sig = sig.replace(parameters=new_params)

    def _lpdf(*args, **kwargs):
        new_args = resolve_kwargs(new_sig, args, kwargs)
        return pyro_lpdf(new_args[0], dist, *new_args[1:])

    _lpdf.__signature__ = new_sig

    return _lpdf


meas_sample = make_dist_sample(meas_dist)
meas_lpdf = make_dist_lpdf(meas_dist)

key = jax.random.PRNGKey(0)
theta = (1.2, 2.3)
x_curr = (-1.0, 2.0)
y_curr = (0.7, -0.8)

jax.jit(meas_sample)(key=key, x_curr=x_curr, theta=theta)
jax.jit(meas_lpdf)(y_curr=y_curr, x_curr=x_curr, theta=theta)

# --- test ---------------------------------------------------------------------


def make_foo(f):
    """
    Create a `foo()` out of `f()`.
    """
    sig = inspect.signature(f)

    def _foo(*args, **kwargs):
        # no copy, positional order, positional order of new_sig
        new_args = resolve_kwargs(
            fun_sig,
            *args,
            **kwargs,
        )
        return bar(new_args[0], f, *new_args[1:])

    return _foo
