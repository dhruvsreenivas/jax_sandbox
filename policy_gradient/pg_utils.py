from common.dataset import TransitionBatch
import jax
import jax.numpy as jnp

@jax.jit
def returns_to_go(batch: TransitionBatch, gamma: float = 1.0) -> jnp.ndarray:
    '''
    Computes returns to go, optionally discounted by gamma.
    
    Rewards are of shape (B,).
    '''
    rewards = batch.rewards
    B = rewards.shape[0]
    if gamma < 1.0:
        discounts = jnp.geomspace(1.0, gamma ** (B - 1), num=B)
        rewards *= discounts
    
    return jnp.cumsum(rewards)