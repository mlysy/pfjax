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


class SDEParams(eqx.Module):
    # eqx.Module wants us to think about NN(x, theta)
    # neural network for drift
    drift_layers: tuple
    # simple diagonal for diffusion
    diff_sd: jax.Array

    def __init__(self, key):
        drift_layer_dims = (2, 8)  # can make this an input
        n_dims = drift_layer_dims[0]  # dimension of SDE
        n_layers = len(drift_layer_dims)  # number of layers
        drift_layer_dims += (n_dims,)  # output_dim = input_dim
        keys = jax.random.split(key, num=n_layers + 1)
        self.layers = tuple()
        for l in range(n_layers):
            self.layers += (
                eqx.nn.Linear(
                    in_features=drift_layers_dims[l],
                    out_features=drift_layers_dims[l + 1],
                    key=keys[l],
                ),
                jax.nn.relu,
            )
        self.diff_sd = jax.random.normal(key=keys[-1], shape=(n_dims,))
        self.diff_sd = jnp.exp(diff_sd)


class NeuralSDE(SDEModel):
    def drift(self, x, theta):
        """
        Define the drift function.
        """
        layers = theta.layers
        for layer in self.layers:
            x = layer(x)
        return x

    def diff(self, x, theta):
        diff_sd = theta.diff_sd
        return diff_sd
