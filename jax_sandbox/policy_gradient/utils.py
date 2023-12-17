import jax
import jax.numpy as jnp
import distrax
import functools
from typing import NamedTuple, Tuple

"""Utils for policy gradient."""

class PGOutput(NamedTuple):
    policy: distrax.Distribution
    values: jax.Array

@functools.partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
def compute_gae(
    rewards: jax.Array,
    masks: jax.Array,
    values: jax.Array,
    gamma: float,
    gae_lam: float,
) -> jax.Array:
    """Compute the GAE advantage estimates from rewards + values."""
    
    assert rewards.shape[0] + 1 == values.shape[0], "Need one more value than reward."
    
    # first compute value diffs and deltas (the advantage at the t'th step)
    value_diffs = gamma * values[1:] * masks[:-1] - values[:-1]
    deltas = rewards + value_diffs # shape [T, n_traj]
    
    # now we compute each GAE
    def gae_loop_body(curr_gae: jax.Array, inp: Tuple[jax.Array, jax.Array]):
        delta, mask = inp
        new_gae = delta + gamma * gae_lam * mask * curr_gae
        
        return new_gae, new_gae
    
    # now we reverse the order of the deltas and masks
    deltas_rev = jnp.flip(deltas, axis=0)
    masks_rev = jnp.flip(masks, axis=0)
    
    _, gaes = jax.lax.scan(
        gae_loop_body, init=0.0, xs=(deltas_rev, masks_rev[:-1])
    )
    gaes = jnp.flip(gaes, axis=0)
    return gaes