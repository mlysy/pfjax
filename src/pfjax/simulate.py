import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from jax import random
from jax import lax
from pfjax.utils import *


def simulate(model, key, n_obs, x_init, theta):
    """
    Simulate data from the state-space model.

    Args:
        model: Object specifying the state-space model having the following methods.

            - `state_sim : (key, x_prev, theta) -> x_curr`: Sample from the state model.
            - `meas_sim : (key, x_curr, theta) -> y_curr`: Sample from the measurement model.

        key: PRNG key.
        n_obs: Number of observations to generate.
        x_init: Initial state value at time `t = 0`.
        theta: Parameter value.

    Returns:
        Tuple: 

        - **y_meas** - The sequence of measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.

        - **x_state** - The sequence of state variables `x_state = (x_0, ..., x_T)`, where `T = n_obs-1`.
    """
    # lax.scan setup
    # scan function
    def fun(carry, x):
        key, *subkeys = random.split(carry["key"], num=3)
        x_state = model.state_sample(subkeys[0], carry["x_state"], theta)
        y_meas = model.meas_sample(subkeys[1], x_state, theta)
        res = {"y_meas": y_meas, "x_state": x_state, "key": key}
        return res, res
    # scan initial value
    key, subkey = random.split(key)
    init = {
        "y_meas": model.meas_sample(subkey, x_init, theta),
        "x_state": x_init,
        "key": key
    }
    # scan itself
    last, full = lax.scan(fun, init, jnp.arange(n_obs-1))
    # append initial values
    x_state = tree_append_first(full["x_state"], first=init["x_state"])
    y_meas = tree_append_first(full["y_meas"], first=init["y_meas"])
    # x_state = jnp.append(jnp.expand_dims(init["x_state"], axis=0),
    #                      full["x_state"], axis=0)
    # y_meas = jnp.append(jnp.expand_dims(init["y_meas"], axis=0),
    #                     full["y_meas"], axis=0)
    return y_meas, x_state
