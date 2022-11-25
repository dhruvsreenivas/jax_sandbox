import jax
import jax.numpy as jnp
import haiku as hk
import distrax
from common.utils import *

from typing import List

'''Various neural network architectures for use in this repository.'''

class MLP(hk.Module):
    '''Standard MLP, with optional layer norm after first layer processing.'''
    def __init__(self,
                 hidden_sizes: List,
                 act: str = 'relu',
                 activate_final: bool = False,
                 use_ln: bool = False):
        super().__init__()
        self._hidden_sizes = hidden_sizes
        self._activation = activation(act)
        self._activate_final = activate_final
        self._use_ln = use_ln
        
    def __call__(self, x):
        x = hk.Linear(self._hidden_sizes[0])(x)
        if self._use_ln:
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            x = jax.lax.tanh(x)
        
        net = hk.nets.MLP(
            self._hidden_sizes[1:],
            activation=self._activation,
            activate_final=self._activate_final
        )
        return net(x)

class ConvTorso(hk.Module):
    def __init__(self,
                 out_channels: List,
                 kernels: List,
                 strides: List,
                 act: str = 'relu',
                 activate_final: bool = False):
        super().__init__()
        assert len(out_channels) == len(kernels) == len(strides), 'not the same amount of channels, kernel shapes and sizes.'
        self._channels = out_channels
        self._kernels = kernels
        self._strides = strides
        self._activation = activation(act)
        self._activate_final = activate_final
        
    def __call__(self, x):
        for channel, kernel, stride in zip(self._channels[:-1], self._kernels[:-1], self._strides[:-1]):
            x = hk.Conv2D(channel, kernel_shape=kernel, stride=stride)
            x = self._activation(x)
            
        x = hk.Conv2D(self._channels[-1], self._kernels[-1], self._strides[-1])
        if self._activate_final:
            x = self._activation(x)
        
        return x
    
class PolicyValueHead(hk.Module):
    '''Policy/value head. Discrete setup.'''
    def __init__(self, out_dim: int):
        super().__init__()
        self._out_dim = out_dim
    
    def __call__(self, x):
        policy_out = hk.Linear(self._out_dim)(x)
        value_out = hk.Linear(1)(x)
        
        return policy_out, value_out
    
class LinearCategorical(hk.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self._out_dim = out_dim
    
    def __call__(self, x):
        out = hk.Linear(self._out_dim)(x)
        dist = distrax.Categorical(logits=out)
        return dist
    
class LinearGaussian(hk.Module):
    def __init__(self, out_dim: int, softplus: bool = False, min_std: float = 0.1):
        super().__init__()
        self._out_dim = out_dim
        self._softplus = softplus
        self._min_std = min_std
    
    def __call__(self, x):
        out = hk.Linear(2 * self._out_dim)(x)
        mean, log_std = jnp.split(out, 2, -1)
        if self._softplus:
            std = jax.nn.softplus(log_std) + self._min_std
        else:
            std = jnp.exp(log_std)
        
        dist = distrax.Independent(distrax.Normal(mean, std), reinterpreted_batch_ndims=1)
        return dist
