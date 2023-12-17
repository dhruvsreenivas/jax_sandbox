from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import functools
import math

from typing import Sequence, Callable, Optional, Union, Tuple


"""General deep learning architectures for reinforcement learning."""

PaddingLike = Union[str, int, Sequence[Union[int, Tuple[int, int]]]]
LaxPadding = Union[str, Sequence[Tuple[int, int]]]


def canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding:
    """Canonicalizes conv padding to a jax.lax supported format. Stolen from Flax source code."""
    if isinstance(padding, str):
        return padding
    if isinstance(padding, int):
        return [(padding, padding)] * rank
    if isinstance(padding, Sequence) and len(padding) == rank:
        new_pad = []
        for p in padding:
            if isinstance(p, int):
                new_pad.append((p, p))
            elif isinstance(p, tuple) and len(p) == 2:
                new_pad.append(p)
            else:
                break
        if len(new_pad) == rank:
            return new_pad
    raise ValueError(
        f'Invalid padding format: {padding}, should be str, int,'
        f' or a sequence of len {rank} where each element is an'
        ' int or pair of ints.'
    )
    
    
def _conv_dimension_numbers(input_shape):
    """Computes the dimension numbers based on the input shape. Stolen from Flax source code."""
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


# ======================== Single layer modifications. ========================


class NoisyDense(nn.Dense):
    """Dense layer with noisy weights. Used in Noisy Networks for Exploration."""
    
    noise_fn: Optional[Callable[[jax.Array], jax.Array]] = None
    
    @nn.compact
    def __call__(self, x: jax.Array, rng: Optional[jax.Array] = None) -> jax.Array:
        
        # initialize trainable parameters
        mu_w = self.param(
            "mu_w", self.kernel_init, (jnp.shape(x)[-1], self.features), self.param_dtype
        )   
        sigma_w = self.param(
            "sigma_w", self.kernel_init, (jnp.shape(x)[-1], self.features)
        )
        
        if self.use_bias:
            mu_b = self.param(
                "mu_b", self.bias_init, (self.features,), self.param_dtype
            )
            sigma_b = self.param(
                "sigma_b", self.bias_init, (self.features,)
            )
        
        # initialize noise
        if rng is None:
            rng = self.make_rng("noise")
            
        mu_rng, sigma_rng = jax.random.split(rng)
        
        if self.noise_fn is None:
            noise_w = jax.random.normal(
                mu_rng, (jnp.shape(x)[-1], self.features), self.param_dtype
            )
            noise_b = jax.random.normal(
                sigma_rng, (self.features,), self.param_dtype
            )
        else:
            eps_i = jax.random.normal(
                mu_rng, (jnp.shape(x)[-1],), self.param_dtype
            )
            eps_j = jax.random.normal(
                sigma_rng, (self.features,), self.param_dtype
            )

            noise_b = self.noise_fn(eps_j) # f(\epsilon_j), shape [features]
            noise_w = jnp.outer(self.noise_fn(eps_i), noise_b) # shape [in_features, features]
            
        if self.dot_general_cls is not None:
            dot_general = self.dot_general_cls()
        elif self.dot_general is not None:
            dot_general = self.dot_general
        else:
            dot_general = jax.lax.dot_general
            
        y = dot_general(
            x, mu_w + sigma_w * noise_w,
            (((x.ndim - 1,), (0,)), ((), ())),
            precision=self.precision
        )
        if self.use_bias:
            bias = mu_b + sigma_b * noise_b
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        
        return y
    
    
class NormedDense(nn.Dense):
    """Normed dense layer, used in PPG's ResNet architecture."""
    
    scale: float = 1.0
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        kernel = self.param(
            "kernel", self.kernel_init, (jnp.shape(x)[-1], self.features), self.param_dtype
        )
        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
            
        # normalize kernel data
        kernel *= self.scale / jnp.linalg.norm(kernel, axis=1, keepdims=True)
        
        if self.dot_general_cls is not None:
            dot_general = self.dot_general_cls()
        elif self.dot_general is not None:
            dot_general = self.dot_general
        else:
            dot_general = jax.lax.dot_general
            
        y = dot_general(
            x, kernel,
            (((x.ndim - 1,), (0,)), ((), ())),
            precision=self.precision
        )
        if self.use_bias:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        
        return y
    
    
class NormedConv(nn.Conv):
    """Layer normed convolutional layer, used in PPG's ResNet architecture."""
    
    scale: float = 1.0
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        
        kernel_size = tuple(self.kernel_size)
        batch_dims = x.ndim - len(kernel_size) - 1
        
        if batch_dims != 1:
            input_batch_shape = x.shape[:batch_dims]
            total_batch_size = int(np.prod(input_batch_shape))
            flat_input_shape = (total_batch_size,) + x.shape[batch_dims:]
            x = jnp.reshape(x, flat_input_shape)
        
        # create the weights
        in_features = jnp.shape(x)[-1]
        assert in_features % self.feature_group_count == 0
        
        kernel_shape = kernel_size + (
            in_features // self.feature_group_count,
            self.features
        )
        kernel = self.param(
            "kernel", self.kernel_init, kernel_shape, self.param_dtype
        )
        
        # normalize the kernel here
        kernel *= self.scale / jnp.linalg.norm(kernel, axis=(0, 1, 2), keepdims=True)
        
        if self.mask is not None:
            kernel *= self.mask
        
        # create the bias
        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None
            
        if self.conv_general_dilated_cls is not None:
            conv_general_dilated = self.conv_general_dilated_cls()
        elif self.conv_general_dilated is not None:
            conv_general_dilated = self.conv_general_dilated
        else:
            conv_general_dilated = jax.lax.conv_general_dilated
            
        def maybe_broadcast(
            x: Optional[Union[int, Sequence[int]]]
        ) -> Tuple[int, ...]:
            if x is None:
                # backward compatibility with using None as sentinel for
                # broadcast 1
                x = 1
            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return tuple(x)
        
        strides = maybe_broadcast(self.strides)
        padding = canonicalize_padding(self.padding)
        input_dilation = maybe_broadcast(self.input_dilation)
        kernel_dilation = maybe_broadcast(self.kernel_dilation)
        num_dimensions = _conv_dimension_numbers(x.shape)
            
        y = conv_general_dilated(
            x, kernel, strides, padding, lhs_dilation=input_dilation,
            rhs_dilation=kernel_dilation, dimension_numbers=num_dimensions,
            feature_group_count=self.feature_group_count, precision=self.precision,
        )
        
        if self.use_bias:
            bias = jnp.reshape(bias, (1,) * (y.ndim - bias.ndim) + bias.shape)
            y += bias
            
        # unflatten now
        if batch_dims != 1:
            output_shape = input_batch_shape + y.shape[1:]
            y = jnp.reshape(y, output_shape)
            
        return y
        

# ======================== Multilayer architectures. ========================
            

class MLP(nn.Module):
    """Feedforward neural network."""
    
    hidden_sizes: Sequence[int]
    output_size: int
    w_init: Callable[[jax.Array], jax.Array] = nn.initializers.orthogonal(scale=math.sqrt(2))
    activation: Callable[[jax.Array], jax.Array] = nn.relu
    final_activation: Optional[Callable[[jax.Array], jax.Array]] = None
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        
        for hidden_size in self.hidden_sizes:
            x = nn.Dense(
                hidden_size,
                kernel_init=self.w_init
            )(x)
            x = self.activation(x)
            
        x = nn.Dense(
            self.output_size, kernel_init=self.w_init
        )(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        
        return x
    
    
class ConvStack(nn.Module):
    """2D convolutional stack."""
    
    channels: Sequence[int]
    kernel_sizes: Sequence[int]
    strides: Sequence[int]
    w_init: Callable[[jax.Array], jax.Array] = nn.initializers.orthogonal(scale=math.sqrt(2))
    activation: Callable[[jax.Array], jax.Array] = nn.relu
    
    flatten_at_end: bool = True
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        batch_dims = x[:-3].ndim
        
        assert len(self.channels) == len(self.kernel_sizes) == len(self.strides), "Number of channels, number of kernels, and number of strides must all be equal."
        
        for ch, kernel_size, stride in zip(self.channels, self.kernel_sizes, self.strides):
            x = nn.Conv(
                ch, kernel_size=(kernel_size, kernel_size), strides=(stride, stride),
                padding="VALID", kernel_init=self.w_init,
            )(x)
            x = self.activation(x)
            
        if self.flatten_at_end:
            x = jnp.reshape(x, (x[:batch_dims], -1))
            
        return x
    
    
class ResNetBlock(nn.Module):
    """ResNet block used in PPG."""
    
    channels: int
    use_normed_conv: bool = True
    use_batch_norm: bool = True
    norm_scale: float = 1.0
    w_init: Callable[[jax.Array], jax.Array] = nn.initializers.orthogonal(scale=math.sqrt(2))
    activation: Callable[[jax.Array], jax.Array] = nn.relu
    
    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        residual = x
        
        # go through residual block
        conv_cls = functools.partial(NormedConv, scale=math.sqrt(self.norm_scale)) if self.use_normed_conv else nn.Conv
        
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not train)(x)
        
        x = self.activation(x)
        x = conv_cls(
            self.channels, kernel_size=(3, 3), padding="SAME", kernel_init=self.w_init,
        )(x)
        
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not train)(x)
        
        x = self.activation(x)
        x = conv_cls(
            self.channels, kernel_size=(3, 3), padding="SAME", kernel_init=self.w_init,
        )(x)
        
        return x + residual
    
    
class ResNetDownStack(nn.Module):
    """Downsample stack used in PPG, similar to IMPALA architecture."""
    
    channels: int
    use_normed_conv: bool = True
    use_batch_norm: bool = True
    norm_scale: float = 1.0
    activation: Callable[[jax.Array], jax.Array] = nn.relu
    w_init: Callable[[jax.Array], jax.Array] = nn.initializers.orthogonal(scale=math.sqrt(2))
    use_max_pool: bool = True
    num_blocks: int = 2
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        conv_cls = functools.partial(NormedConv, scale=math.sqrt(self.norm_scale)) if self.use_normed_conv else nn.Conv
        
        x = conv_cls(
            self.channels, (3, 3), padding="SAME", name="init_conv", kernel_init=self.w_init,
        )(x)
        if self.use_max_pool:
            x = nn.max_pool(
                x, window_shape=(3, 3), strides=(2, 2)
            )(x)
            
        for i in range(self.num_blocks):
            x = ResNetBlock(
                self.channels, self.use_normed_conv, self.use_batch_norm, self.norm_scale,
                self.activation, w_init=self.w_init, name=f"resblock_{i}",
            )(x)
            
        return x