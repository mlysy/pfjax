# tutorial from:
# https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html

from typing import Any, Callable
import jax
from jax.scipy.special import logsumexp
from jax.nn import one_hot, relu
from jax import lax
import jax.numpy as jnp
# Use 8 CPU devices
# need to do this before doing anything with jax:
# https://github.com/google/jax/issues/6887
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'


def predict(w1, w2, images):
    hiddens = relu(jnp.dot(images, w1))
    logits = jnp.dot(hiddens, w2)
    return logits - logsumexp(logits, axis=1, keepdims=True)


def loss(w1, w2, images, labels):
    predictions = predict(w1, w2, images)
    targets = one_hot(labels, predictions.shape[-1])
    losses = jnp.sum(targets * predictions, axis=1)
    return -jnp.mean(losses, axis=0)


w1 = jnp.zeros((784, 512))
w2 = jnp.zeros((512, 10))
images = jnp.zeros((128, 784))
labels = jnp.zeros(128, dtype=jnp.int32)

loss(w1, w2, images, labels)


class ArrayType:
    def __getitem__(self, idx):
        return Any


f32 = ArrayType()
i32 = ArrayType()

breakpoint()
