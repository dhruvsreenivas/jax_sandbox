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

def get_activation(act):
    if act in ACTIVATIONS.keys():
        return ACTIVATIONS[act]
    raise ValueError('Activation not available.')

def get_opt_class(opt_name) -> optax.GradientTransformation:
    if opt_name in OPTIMIZERS.keys():
        return OPTIMIZERS[opt_name]
    raise ValueError('Optimizer not available.')

def ema_update(online_params, target_params, tau=0.9):
    '''Returns new params!'''
    return optax.incremental_update(online_params, target_params, 1.0 - tau)