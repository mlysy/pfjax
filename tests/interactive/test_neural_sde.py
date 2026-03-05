import equinox as eqx
import jax
import jax.numpy as jnp
import pfjax

# define drift and diffusion using equinox modules

# equinox paradigm:
# f(x, theta) is defined as follows:
#
# class f(eqx.Module):
#    theta: PyTree
#    def __call__(x):
#        return f_impl(x, theta)


class DriftDiff(eqx.Module):
    # eqx.Module wants us to think about NN(x, theta)
    # neural network for drift
    layers: tuple
    # simple diagonal for diffusion
    diff_sd: jax.Array

    def __init__(self, key):
        key1, key2 = jax.random.split(key)
        self.layers = (
            eqx.nn.Linear(2, 8, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(8, 2, key=key2),
        )

    def drift(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def diff(self, x):
        return diff_sd


class NeuralSDE(SDEModel):
    def drift(self, x, theta):
        """
        Define the drift function.

        theta: object of class DriftDiff
        """
        return theta.drift(x)

    def diff(self, x, theta):
        return theta.diff(x)


model = NeuralSDE(
    dt,
    n_res,
    diag,
    drift_net,
    diff_net,
)
