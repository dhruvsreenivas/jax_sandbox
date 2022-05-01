'''General utils folder.'''
import optax
import jax.numpy as jnp
import jax.nn as nn

ACTIVATIONS = {
    'relu': nn.relu,
    'elu': nn.elu,
    'tanh': jnp.tanh, # not in jax.nn
    'leaky_relu': nn.leaky_relu,
    'sigmoid': nn.sigmoid
}

OPTIMIZERS = {
    'adam': optax.adam,
    'sgd': optax.sgd,
    'rmsprop': optax.rmsprop
}

def get_opt_class(opt_name):
    return OPTIMIZERS[opt_name]