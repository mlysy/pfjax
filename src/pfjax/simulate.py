"""
Simulate trajectories from a state space model.

The API requires the user to define a model class with the following methods:

- `state_sample()`
- `meas_sample()`

The provided function is:

- `simulate()`
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import lax


def simulate_for(model, key, n_obs, x_init, theta):
    """
    Simulate data from the state-space model.

    **FIXME:** This is the testing version which uses a for-loop.  This should be put in a separate class in a `test` subfolder.

    Args:
        model: Object specifying the state-space model.
        key: PRNG key.
        n_obs: Number of observations to generate.
        x_init: Initial state value at time `t = 0`.
        theta: Parameter value.

    Returns:
        y_meas: The sequence of measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        x_state: The sequence of state variables `x_state = (x_0, ..., x_T)`, where `T = n_obs-1`.
    """
    y_meas = jnp.zeros((n_obs, model.n_meas))
    x_state = jnp.zeros((n_obs, model.n_state))
    x_state = x_state.at[0].set(x_init)
    # initial observation
    key, subkey = random.split(key)
    y_meas = y_meas.at[0].set(model.meas_sample(subkey, x_init, theta))
    for t in range(1, n_obs):
        key, *subkeys = random.split(key, num=3)
        x_state = x_state.at[t].set(
            model.state_sample(subkeys[0], x_state[t-1], theta)
        )
        y_meas = y_meas.at[t].set(
            model.meas_sample(subkeys[1], x_state[t], theta)
        )
    return y_meas, x_state

# @partial(jax.jit, static_argnums=0)


def simulate(model, key, n_obs, x_init, theta):
    """
    Simulate data from the state-space model.

    Args:
        model: Object specifying the state-space model.
        key: PRNG key.
        n_obs: Number of observations to generate.
        x_init: Initial state value at time `t = 0`.
        theta: Parameter value.

    Returns:
        y_meas: The sequence of measurement variables `y_meas = (y_0, ..., y_T)`, where `T = n_obs-1`.
        x_state: The sequence of state variables `x_state = (x_0, ..., x_T)`, where `T = n_obs-1`.
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
    last, full = lax.scan(fun, init, jnp.arange(1, n_obs))
    # append initial values
    x_state = jnp.append(jnp.expand_dims(init["x_state"], axis=0),
                         full["x_state"], axis=0)
    y_meas = jnp.append(jnp.expand_dims(init["y_meas"], axis=0),
                        full["y_meas"], axis=0)
    return y_meas, x_state
