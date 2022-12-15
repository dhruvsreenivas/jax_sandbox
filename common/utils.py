'''General utils file.'''
import haiku as hk
import optax
import jax.numpy as jnp
import jax.nn as nn
import jax.lax as lax
from jax.random import PRNGKey
import distrax

ACTIVATIONS = {
    'relu': nn.relu,
    'elu': nn.elu,
    'tanh': lax.tanh, # not in jax.nn
    'leaky_relu': nn.leaky_relu,
    'sigmoid': nn.sigmoid,
    'identity': lambda x: x
}

OPTIMIZERS = {
    'adam': optax.adam,
    'adamw': optax.adamw,
    'sgd': optax.sgd,
    'rmsprop': optax.rmsprop
}

def activation(act: str):
    if act in ACTIVATIONS.keys():
        return ACTIVATIONS[act]
    raise ValueError('Activation not available.')

def opt_class(opt_name: str) -> optax.GradientTransformation:
    if opt_name in OPTIMIZERS.keys():
        return OPTIMIZERS[opt_name]
    raise ValueError('Optimizer not available.')

def update_target(online_params: hk.Params, target_params: hk.Params, tau: float = 0.9) -> hk.Params:
    '''Returns new params!'''
    return optax.incremental_update(online_params, target_params, tau)

def batched_zeros_like(shape) -> jnp.ndarray:
    if type(shape) == int:
        return jnp.zeros((1, shape))
    return jnp.zeros((1,) + tuple(shape))

def sample_multiple(dist: distrax.Distribution, key: PRNGKey, n: int, log_prob: bool = True):
    if log_prob:
        return dist._sample_n_and_log_prob(key, n)
    return dist._sample_n(key, n)