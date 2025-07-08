# --- using numpyro to create _sim and _lpdf -----------------------------------

# Works fine for random quantity as a JAX array.
# Not sure how to automate this when its a PyTree.

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


# example


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


x_prev = 5.1
theta = 1.7
x_curr = {"x": 1.0, "y": 2.0}

jax.jit(pyro_lpdf, static_argnums=1)(
    data=x_curr, dist=pyro_dist, x_prev=x_prev, theta=theta
)

jax.jit(pyro_sample, static_argnums=1)(
    key=key, dist=pyro_dist, x_prev=x_prev, theta=theta
)
