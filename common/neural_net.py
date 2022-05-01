import jax.numpy as jnp
import haiku as hk
import distrax
from utils import ACTIVATIONS
import numpy as np

'''Various neural network architectures for use in this repository.'''

class MLP(hk.Module):
    '''Standard MLP. Can be plugged in anywhere.'''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.output_size = cfg.output_size
        self.n_hidden_layers = cfg.n_hidden_layers
        
        self.activation = ACTIVATIONS[cfg.activation]
        
    def __call__(self, input):
        assert jnp.ndim(input) > 0, 'Input must be a (1+)D JAX numpy array'
        
        if jnp.ndim(input) > 1:
            input = hk.Flatten()(input)
        
        for _ in range(self.n_hidden_layers):
            input = hk.Linear(self.hidden_size)(input)
            input = self.activation(input)
        
        output = hk.Linear(self.output_size)(input)
        return output

    def output_dim(self, in_shape):
        return self.output_size
    
class ConvBackbone(hk.Module):
    '''Convolutional neural network backbone. Taken from DQN'''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.out_channels = cfg.out_channels
        self.kernel_sizes = cfg.kernel_sizes
        self.strides = cfg.strides
        
        self.activation = ACTIVATIONS[cfg.activation]
        
        layers = []
        for out_channel, kernel_size, stride in zip(cfg.out_channels, cfg.kernel_sizes, cfg.strides):
            layers.append(hk.Conv2D(out_channel, kernel_shape=kernel_size, stride=stride, padding='VALID'))
            layers.append(self.activation)
        
        self.trunk = hk.Sequential(layers)
        
    def __call__(self, input):
        '''Output will not be flattened.'''
        assert jnp.ndim(input) in [3, 4], 'Not valid 2D input'
        return self.trunk(input)

    def output_dim(self, in_shape):
        shape = in_shape # assumed, potentially channel dim is at end?
        for out_channel, kernel_size, stride in zip(self.out_channels, self.kernel_sizes, self.strides):
            hin = shape[1]
            win = shape[2]
            
            hout = np.floor((hin + 2 * 0 - 1 * (kernel_size - 1) - 1) / stride + 1)
            wout = np.floor((win + 2 * 0 - 1 * (kernel_size - 1) - 1) / stride + 1)
            
            shape = (out_channel, hout, wout)
        
        return np.prod(shape)
    
class DiscreteQNetwork(hk.Module):
    '''Discrete Q function network.'''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.img_input = cfg.img_input
        
    def __call__(self, states):
        if self.img_input:
            features = ConvBackbone(self.cfg.conv_args)(states)
            features = hk.Flatten()(features)
        else:
            features = states
        
        return MLP(self.cfg.mlp_args)(features)
    
class ContinuousQNetwork(hk.Module):
    '''Continuous Q function network.'''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.img_input = cfg.img_input
    
    def __call__(self, states, actions):
        if self.img_input:
            features = ConvBackbone(self.cfg.conv_args)(states)
            features = hk.Flatten()(features)
        else:
            features = states
        
        sa = jnp.concatenate((features, actions), axis=1)
        return MLP(self.cfg.mlp_args)(sa)

class VNetwork(hk.Module):
    '''Traditional value network (s -> V(s)).'''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.img_input = cfg.img_input
        
        if cfg.img_input:
            self.trunk = ConvBackbone(cfg.conv_args)
        
    def __call__(self, states):
        if self.img_input:
            features = ConvBackbone(self.cfg.conv_args)(states)
            features = hk.Flatten()(features)
        else:
            features = states
        
        return MLP(self.cfg.mlp_args)(features)
    
class DiscretePolicy(hk.Module):
    '''Discrete policy network.'''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.img_input = cfg.img_input

        self.policy = MLP(cfg.mlp_args) # kwargs contains linear stuff, including output (action) dim
        
    def __call__(self, states):
        if self.img_input:
            features = ConvBackbone(self.cfg.conv_args)(states)
            features = hk.Flatten()(features)
        else:
            features = states
        
        logits = MLP(self.cfg.mlp_args)(features)
        return distrax.Transformed(distrax.Categorical(logits=logits))

    def feature_output_dim(self, in_shape):
        if self.img_input:
            return ConvBackbone(self.cfg.conv_args).output_dim(in_shape)
        else:
            return in_shape
    
class ContinuousPolicy(hk.Module):
    '''Continuous policy network.'''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.img_input = cfg.img_input
    
    def __call__(self, states):
        if self.img_input:
            features = ConvBackbone(self.cfg.conv_args)(states)
            features = hk.Flatten()(features)
        else:
            features = states
        
        # TODO: make sure the output is 2 * action_dim
        mean, log_std = jnp.split(MLP(self.cfg.mlp_args)(features), 2, -1)
        std = jnp.exp(log_std)
        
        return distrax.Transformed(distrax.Normal(loc=mean, scale=std))
    
    def feature_output_dim(self, in_shape):
        if self.img_input:
            return ConvBackbone(self.cfg.conv_args).output_dim(in_shape)
        else:
            return in_shape