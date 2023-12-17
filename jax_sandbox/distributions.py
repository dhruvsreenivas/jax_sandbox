import jax
import jax.numpy as jnp
import distrax
from typing import Optional

"""Additional distributions in JAX."""


class TanhMultivariateNormalDiag(distrax.Transformed):
    """Multivariate normal diagonal Gaussian distribution, with tanh squashing.
    
    See https://github.com/ikostrikov/jaxrl2/blob/main/jaxrl2/networks/normal_tanh_policy.py#L11.
    """
    
    def __init__(
        self, loc: jax.Array, scale: jax.Array,
        low: Optional[jax.Array] = None, high: Optional[jax.Array] = None,
    ) -> None:
        
        dist = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale)
        
        layers = []
        if not (low is None or high is None):
            
            def rescale_from_tanh(x: jax.Array) -> jax.Array:
                x = (x + 1) / 2
                return x * (high - low) + low
            
            def forward_log_det_jacobian(x: jax.Array) -> jax.Array:
                high_ = jnp.broadcast_to(high, x.shape)
                low_ = jnp.broadcast_to(low, x.shape)
                
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), axis=-1)
            
            layers.append(
                distrax.Lambda(
                    rescale_from_tanh,
                    forward_log_det_jacobian=forward_log_det_jacobian,
                    event_ndims_in=1,
                    event_ndims_out=1
                )
            )
            
        layers.append(distrax.Block(distrax.Tanh(), 1))
        
        # create the bijector
        bijector = distrax.Chain(layers)
        
        # now define the transformed distribution
        super().__init__(dist, bijector)
        
        
    def mode(self) -> jax.Array:
        return self.bijector.forward(self.distribution.mode())