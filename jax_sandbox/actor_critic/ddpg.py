import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax
from typing import Sequence, Callable, Tuple
from ml_collections import ConfigDict
import functools

from jax_sandbox.nets import MLP
from jax_sandbox.base_learner import Learner
from jax_sandbox.dataset import Batch
from jax_sandbox.utils import MetricsDict, TrainStateWithTarget


class Actor(nn.Module):
    """Actor for DDPG."""
    
    action_dim: int
    hidden_sizes: Sequence[int]
    w_init: Callable[[jax.Array], jax.Array] = nn.initializers.he_uniform
    activation: Callable[[jax.Array], jax.Array] = nn.relu
    max_action: float = 1.0
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        
        x = MLP(
            hidden_sizes=self.hidden_sizes,
            output_size=self.action_dim,
            w_init=self.w_init,
            activation=self.activation,
        )(x)
        
        return jnp.tanh(x) * self.max_action
    
    
class Critic(nn.Module):
    """Critic for DDPG."""
    
    action_dim: int
    hidden_sizes: Sequence[int]
    w_init: Callable[[jax.Array], jax.Array] = nn.initializers.he_uniform
    activation: Callable[[jax.Array], jax.Array] = nn.relu
    
    @nn.compact
    def __call__(self, x: jax.Array, a: jax.Array) -> jax.Array:
        assert a.shape[-1] == self.action_dim, f"Expected shape {self.action_dim} but got {a.shape[-1]}."
        
        if self.hidden_sizes == ():
            xa = jnp.concatenate([x, a], axis=-1)
            return nn.Dense(
                1, kernel_init=self.w_init
            )(xa)
        else:
            x = nn.Dense(
                self.hidden_sizes[0], kernel_init=self.w_init
            )(x)
            x = self.activation(x)
            
            xa = jnp.concatenate([x, a], axis=-1)
            xa = MLP(
                hidden_sizes=self.hidden_sizes[1:],
                output_size=1,
                w_init=self.w_init,
                activation=self.activation,
            )(xa)
            return xa
    
# ========================= Action + loss functions and update steps =========================

@jax.jit
def act(
    actor_state: TrainStateWithTarget, observation: jax.Array,
) -> jax.Array:
    """Jitted function to act."""
    
    action = actor_state.apply_fn(
        {"params": actor_state.params}, observation
    )
    return action
    

@functools.partial(jax.jit, static_argnames=("gamma", "tau"))
def update_critic(
    critic_state: TrainStateWithTarget,
    actor_state: TrainStateWithTarget,
    batch: Batch,
    gamma: float,
    tau: float,
) -> Tuple[TrainStateWithTarget, MetricsDict]:
    """Critic update step."""
    
    def critic_loss_fn(params: flax.core.FrozenDict) -> Tuple[jax.Array, MetricsDict]:
        """Critic loss function."""
        
        # compute target Q values
        next_actions = actor_state.apply_fn(
            {"params": actor_state.target_params}, batch.next_observations
        )
        target_q = critic_state.apply_fn(
            {"params": critic_state.target_params}, batch.next_observations, next_actions
        )
        target_q = batch.rewards + gamma * batch.not_dones * target_q
        
        # compute current Q value estimates
        curr_q = critic_state.apply_fn(
            {"params": params}, batch.observations, batch.actions
        )
        
        # compute loss
        loss = jnp.mean(jnp.square(curr_q - target_q))
        
        return loss, {"critic_loss": loss, "curr_q": jnp.mean(curr_q)}
    
    # do optimization step
    grads, metrics = jax.grad(critic_loss_fn, has_aux=True)(critic_state.params)
    new_critic_state = critic_state.apply_gradients(grads=grads)
    
    # do target update
    new_target_params = optax.incremental_update(
        new_critic_state.params, critic_state.target_params, tau
    )
    new_critic_state = new_critic_state.replace(
        target_params=new_target_params
    )
    
    return new_critic_state, metrics


@functools.partial(jax.jit, static_argnames=("tau",))
def update_actor(
    actor_state: TrainStateWithTarget,
    critic_state: TrainStateWithTarget,
    batch: Batch,
    tau: float,
) -> Tuple[TrainStateWithTarget, MetricsDict]:
    """Actor update step."""
    
    def actor_loss_fn(params: flax.core.FrozenDict) -> Tuple[jax.Array, MetricsDict]:
        """Actor loss function."""
        
        # compute Q values
        actions = actor_state.apply_fn(
            {"params": params}, batch.observations
        )
        qs = critic_state.apply_fn(
            {"params": critic_state.params}, batch.observations, actions
        )
        
        # compute loss
        loss = -jnp.mean(qs)
        return loss, {"actor_loss": loss}
    
    # do optimization
    grads, metrics = jax.grad(actor_loss_fn, has_aux=True)(actor_state.params)
    new_actor_state = actor_state.apply_gradients(grads=grads)
    
    # do EMA update
    new_target_params = optax.incremental_update(
        new_actor_state.params, actor_state.target_params, tau
    )
    new_actor_state = new_actor_state.replace(
        target_params=new_target_params
    )
    
    return new_actor_state, metrics

# ========================= Agent construction. =========================

class DDPG(Learner):
    """Deep determinstic policy gradient algorithm."""
    
    def __init__(self, config: ConfigDict):
        self._config = config
        
        # set seeds
        rng = jax.random.PRNGKey(config.seed)
        actor_rng, critic_rng = jax.random.split(rng)
        
        # initialize actor
        w_init = getattr(nn.initializers, config.w_init, nn.initializers.he_uniform)
        activation = getattr(nn, config.activation, nn.relu)
        actor = Actor(
            config.action_dim, config.hidden_sizes,
            w_init=w_init, activation=activation,
            max_action=config.max_action, name="ddpg_actor"
        )
        actor_params = target_actor_params = actor.init(
            actor_rng, jnp.zeros((1, config.observation_dim))
        )["params"]
        actor_optimizer = optax.adam(config.actor_lr)
        
        self._actor_state = TrainStateWithTarget.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=actor_optimizer,
            target_params=target_actor_params,
        )
        
        # initialize critic
        critic = Critic(
            config.action_dim, config.hidden_sizes,
            w_init, activation, name="ddpg_critic"
        )
        critic_params = target_critic_params = critic.init(
            critic_rng, jnp.zeros((1, config.observation_dim)), jnp.zeros((1, config.action_dim))
        )["params"]
        critic_optimizer = optax.adam(config.critic_lr)
        
        self._critic_state = TrainStateWithTarget.create(
            apply_fn=critic.apply,
            params=critic_params,
            tx=critic_optimizer,
            target_params=target_critic_params,
        )
        
    def act(self, observation: jax.Array, eval: bool = False) -> jax.Array:
        del eval
        
        return act(self._actor_state, observation)
    
    def update(self, batch: Batch, step: int) -> MetricsDict:
        del step
        
        # first update critic
        self._critic_state, critic_info = update_critic(
            self._critic_state, self._actor_state,
            batch, self._config.gamma, self._config.tau
        )
        
        # then update actor
        self._actor_state, actor_info = update_actor(
            self._actor_state, self._critic_state,
            batch, self._config.tau
        )
        
        return {**critic_info, **actor_info}