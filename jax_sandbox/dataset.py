import jax
import numpy as np
from typing import NamedTuple, Sequence, Optional, Tuple, List

class Batch(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    not_dones: jax.Array


class OnPolicyBatch(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    not_dones: jax.Array

    log_probs: jax.Array
    returns_to_go: jax.Array
    
    
class OnPolicyBatchWithLogits(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    not_dones: jax.Array

    log_probs: jax.Array
    returns_to_go: jax.Array
    logits: jax.Array
    

class ReplayBuffer:
    """Off-policy replay buffer."""
    
    def __init__(
        self,
        capacity: int,
        observation_shape: Sequence[int],
        action_dim: int,
    ) -> None:
        
        self._observations = np.empty((capacity, *observation_shape), dtype=np.float32)
        self._actions = np.empty((capacity, action_dim), dtype=np.float32)
        self._rewards = np.empty((capacity,), dtype=np.float32)
        self._next_observations = np.empty((capacity, *observation_shape), dtype=np.float32)
        self._not_dones = np.empty((capacity,), dtype=np.float32)
        
        self._capacity = capacity
        self._size = 0
        self._ptr = 0
        
    def size(self):
        return self._size
    
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: np.ndarray,
        done: np.ndarray
    ) -> None:
        
        self._observations[self._ptr] = observation
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._next_observations[self._ptr] = next_observation
        self._not_dones[self._ptr] = 1.0 - done
        
        self._size = min(self._size + 1, self._capacity)
        self._ptr = (self._ptr + 1) % self._capacity
        
    def sample(self, batch_size: int) -> Batch:
        idxs = np.random.randint(0, self._size, batch_size)
        
        return Batch(
            observations=jax.device_put(self._observations[idxs]),
            actions=jax.device_put(self._actions[idxs]),
            rewards=jax.device_put(self._rewards[idxs]),
            next_observations=jax.device_put(self._next_observations[idxs]),
            not_dones=jax.device_put(self._not_dones[idxs])
        )
        
    def sample_all(self) -> Batch:
        return Batch(
            observations=jax.device_put(self._observations[:self._size]),
            actions=jax.device_put(self._actions[:self._size]),
            rewards=jax.device_put(self._rewards[:self._size]),
            next_observations=jax.device_put(self._next_observations[{self._size}]),
            not_dones=jax.device_put(self._not_dones[:self._size])
        )


class OnPolicyBuffer(ReplayBuffer):
    """On-policy buffer, with additional methods such as resetting."""
    
    def __init__(
        self,
        capacity: int,
        observation_shape: Sequence[int],
        action_dim: int,
        gamma: float,
        gae_lam: Optional[float] = None,
    ):
        super().__init__(capacity, observation_shape, action_dim)
        self._observation_shape = observation_shape
        self._action_dim = action_dim
        
        self._log_probs = np.empty((capacity,), dtype=np.float32)
        self._returns = np.empty((capacity,), dtype=np.float32)
        self._gamma = gamma
        self._gae_lam = gae_lam
        
    def reset(self):
        """Clears and resets the buffer."""
        
        self._observations = np.empty((self._capacity, *self._observation_shape), dtype=np.float32)
        self._actions = np.empty((self._capacity, self._action_dim), dtype=np.float32)
        self._rewards = np.empty((self._capacity,), dtype=np.float32)
        self._next_observations = np.empty((self._capacity, *self._observation_shape), dtype=np.float32)
        self._not_dones = np.empty((self._capacity,), dtype=np.float32)
        self._log_probs = np.empty((self._capacity,), dtype=np.float32)
        self._returns = np.empty((self._capacity,), dtype=np.float32)
        
        self._size = 0
        self._ptr = 0
        
    def compute_returns(self) -> None:
        """Computes discounted returns along the first axis."""
        
        def loop_body(rtg: float, reward: float) -> Tuple[jax.Array, jax.Array]:
            new_rtg = reward + self._gamma * rtg
            return new_rtg, new_rtg
            
        _, returns_to_go = jax.lax.scan(
            loop_body, init=0.0, xs=np.flip(self._rewards)
        )
        self._returns = np.flip(np.asarray(returns_to_go))
        
    def sample(self, batch_size: int) -> Batch:
        """(Possible unused) sampling method."""
        idxs = np.random.randint(0, self._size, batch_size)
        
        return OnPolicyBatch(
            observations=jax.device_put(self._observations[idxs]),
            actions=jax.device_put(self._actions[idxs]),
            rewards=jax.device_put(self._rewards[idxs]),
            next_observations=jax.device_put(self._next_observations[idxs]),
            not_dones=jax.device_put(self._not_dones[idxs]),
            log_probs=jax.device_put(self._log_probs[idxs]),
            returns_to_go=jax.device_put(self._returns[idxs])
        )
        
    def sample_all(self) -> OnPolicyBatch:
        return OnPolicyBatch(
            observations=jax.device_put(self._observations[:self._size]),
            actions=jax.device_put(self._actions[:self._size]),
            rewards=jax.device_put(self._rewards[:self._size]),
            next_observations=jax.device_put(self._next_observations[{self._size}]),
            not_dones=jax.device_put(self._not_dones[:self._size]),
            log_probs=jax.device_put(self._log_probs[:self._size]),
            returns_to_go=jax.device_put(self._returns[:self._size])
        )
        
        
def make_minibatches(batch: OnPolicyBatch, minibatch_size: int) -> List[OnPolicyBatch]:
    """Makes a bunch of minibatches from a given full batch."""
    
    bs = batch.observations.shape[0]
    assert bs % minibatch_size == 0, "Batch size must be divisible by minibatch size."
    
    mbs = []
    for i in range(0, bs, minibatch_size):
        idxs = slice(i, i + minibatch_size)
        
        mbs.append(
            OnPolicyBatch(
                observations=batch.observations[idxs],
                actions=batch.actions[idxs],
                rewards=batch.rewards[idxs],
                next_observations=batch.next_observations[idxs],
                not_dones=batch.not_dones[idxs],
                log_probs=batch.log_probs[idxs],
                returns_to_go=batch.returns_to_go[idxs]
            )
        )
        
    return mbs