import flax
from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
import distrax
from typing import Sequence, Callable, Tuple, Optional
from ml_collections import ConfigDict
import functools

from jax_sandbox.nets import MLP
from jax_sandbox.base_learner import Learner
from jax_sandbox.dataset import Batch
from jax_sandbox.utils import MetricsDict, TrainStateWithTarget
from jax_sandbox.distributions import TanhMultivariateNormalDiag


class Actor(nn.Module):
    """Actor for SAC."""
    
    action_dim: int
    hidden_sizes: Sequence[int]
    w_init: Callable[[jax.Array], jax.Array] = nn.initializers.he_uniform
    activation: Callable[[jax.Array], jax.Array] = nn.relu
    log_std_min: Optional[float] = -20.0
    log_std_max: Optional[float] = 2.0
    low: Optional[float] = None
    high: Optional[float] = None
    
    @nn.compact
    def __call__(self, x: jax.Array) -> distrax.Distribution:
        
        x = MLP(
            hidden_sizes=self.hidden_sizes[:-1],
            output_size=self.hidden_sizes[-1],
            w_init=self.w_init,
            activation=self.activation,
            final_activation=self.activation,
            name="sac_actor_trunk"
        )(x)
        
        mu = nn.Dense(
            self.action_dim, kernel_init=self.w_init,
        )(x)
        logstd = nn.Dense(
            self.action_dim, kernel_init=self.w_init
        )(x)
        
        # clamp logstd
        logstd = jnp.clip(
            logstd, self.log_std_min, self.log_std_max
        )
        
        return TanhMultivariateNormalDiag(
            loc=mu, scale=logstd, low=self.low, high=self.high
        )
        
        
class Critic(nn.Module):
    """SAC twin critic. Same as TD3 critic."""
    
    action_dim: int
    hidden_sizes: Sequence[int]
    w_init: Callable[[jax.Array], jax.Array] = nn.initializers.he_uniform
    activation: Callable[[jax.Array], jax.Array] = nn.relu
    
    @nn.compact
    def __call__(self, x: jax.Array, a: jax.Array) -> Tuple[jax.Array, jax.Array]:
        assert a.shape[-1] == self.action_dim, f"Expected shape {self.action_dim} but got {a.shape[-1]}."
        
        xa = jnp.concatenate([x, a], axis=-1)
        
        q1 = MLP(
            hidden_sizes=self.hidden_sizes,
            output_size=1,
            w_init=self.w_init,
            activation=self.activation,
            name="q1"
        )(xa)
        
        q2 = MLP(
            hidden_sizes=self.hidden_sizes,
            output_size=1,
            w_init=self.w_init,
            activation=self.activation,
            name="q2"
        )(xa)
        
        return q1, q2
    
    
class Temperature(nn.Module):
    """Temperature parameter for SAC."""
    
    initial_value: float = 1.0
    
    @nn.compact
    def __call__(self) -> jax.Array:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_value)),
        )
        return jnp.exp(log_temp)
    
# ========================= Action + loss functions and update steps =========================

@jax.jit
def act(
    actor_state: TrainState, observation: jax.Array, eval: bool, rng: jax.random.PRNGKey
) -> jax.Array:
    """Jitted function to act."""
    
    action_dist = actor_state.apply_fn(
        {"params": actor_state.params}, observation
    )
    return jnp.where(
        eval,
        action_dist.mode(),
        action_dist.sample(seed=rng)
    )


@functools.partial(jax.jit, static_argnames=("gamma", "tau"))
def update_critic(
    critic_state: TrainStateWithTarget,
    actor_state: TrainState,
    temp_state: TrainState,
    batch: Batch,
    gamma: float,
    tau: float,
    rng: jax.random.PRNGKey,
) -> Tuple[TrainStateWithTarget, MetricsDict]:
    """SAC critic update step."""
    
    def critic_loss_fn(params: flax.core.FrozenDict) -> Tuple[jax.Array, MetricsDict]:
        """Critic loss function."""
        
        # compute entropy regularized target Q values
        next_action_dist = actor_state.apply_fn(
            {"params": actor_state.params}, batch.next_observations
        )
        next_actions, next_log_probs = next_action_dist.sample_and_log_prob(seed=rng)
        
        target_q1, target_q2 = critic_state.apply_fn(
            critic_state.target_params, batch.next_observations, next_actions
        )
        target_q = jnp.minimum(target_q1, target_q2)
        
        temp = temp_state.apply_fn({"params": temp_state.params})
        target_q = batch.rewards + gamma * batch.not_dones * (target_q - temp * next_log_probs)
        
        # compute current Q estimates
        curr_q1, curr_q2 = critic_state.apply_fn(
            {"params": params}, batch.observations, batch.actions
        )
        loss = jnp.mean(jnp.square(curr_q1 - target_q)) + jnp.mean(jnp.square(curr_q2 - target_q))
        
        return loss, {"critic_loss": loss, "q1": jnp.mean(curr_q1)}
    
    # do the optimization
    grads, metrics = jax.grad(critic_loss_fn, has_aux=True)(critic_state.params)
    new_critic_state = critic_state.apply_gradients(grads=grads)
    
    # update the target parameters
    new_target_params = optax.incremental_update(
        new_critic_state.params, critic_state.target_params, tau
    )
    new_critic_state = new_critic_state.replace(
        target_params=new_target_params
    )
    
    return new_critic_state, metrics


@jax.jit
def update_actor(
    actor_state: TrainState,
    critic_state: TrainStateWithTarget,
    temp_state: TrainState,
    batch: Batch,
    rng: jax.random.PRNGKey
) -> Tuple[TrainState, MetricsDict]:
    """SAC actor update step."""
    
    def actor_loss_fn(params: flax.core.FrozenDict) -> Tuple[jax.Array, MetricsDict]:
        """Actor loss function."""
        
        action_dist = actor_state.apply_fn(
            {"params": params}, batch.observations
        )
        actions, log_probs = action_dist.sample_and_log_prob(seed=rng)
        
        q1, q2 = critic_state.apply_fn(
            {"params": critic_state.params}, batch.observations, actions
        )
        q = jnp.minimum(q1, q2)
        
        temp = temp_state.apply_fn({"params": temp_state.params})
        q = q - temp * log_probs
        
        # compute loss and return
        loss = -jnp.mean(q)
        return loss, {"actor_loss": loss, "entropy": -jnp.mean(log_probs)}
    
    grads, metrics = jax.grad(actor_loss_fn, has_aux=True)(actor_state.params)
    new_actor_state = actor_state.apply_gradients(grads=grads)
    
    return new_actor_state, metrics

@functools.partial(jax.jit, static_argnames=("target_entropy"))
def update_temperature(
    temp_state: TrainState,
    entropy: jax.Array,
    target_entropy: float,
) -> Tuple[TrainState, MetricsDict]:
    """Temperature update step."""
    
    def temp_loss_fn(params: flax.core.FrozenDict) -> Tuple[jax.Array, MetricsDict]:
        """Temperature loss function."""
        
        temp = temp_state.apply_fn({"params": params})
        loss = temp * (entropy - target_entropy).mean()
        
        return loss, {"temp": temp, "temp_loss": loss}
    
    grads, metrics = jax.grad(temp_loss_fn)(temp_state.params)
    new_temp_state = temp_state.apply_gradients(grads=grads)
    
    return new_temp_state, metrics


class SAC(Learner):
    """Soft actor-critic."""
    
    def __init__(self, config: ConfigDict):
        self._config = config
        
        rng = jax.random.PRNGKey(config.seed)
        self._rng, actor_rng, critic_rng, temp_rng = jax.random.split(rng, 4)
        
        # initialize actor
        w_init = getattr(nn.initializers, config.w_init, nn.initializers.he_uniform)
        activation = getattr(nn, config.activation, nn.relu)
        actor = Actor(
            config.action_dim, config.hidden_sizes, w_init,
            activation, config.log_std_min, config.log_std_max,
            config.low, config.high, name="sac_actor"
        )
        actor_params = actor.init(
            actor_rng, jnp.zeros((1, config.observation_dim))
        )["params"]
        actor_optimizer = optax.adam(config.actor_lr)
        
        self._actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=actor_optimizer,
        )
        
        # initialize critic
        critic = Critic(
            config.action_dim, config.hidden_sizes, w_init,
            activation, name="sac_critic"
        )
        critic_params = target_critic_params = critic.init(
            critic_rng, jnp.zeros((1, config.observation_dim)), jnp.zeros((1, config.action_dim))
        )["params"]
        critic_optimizer = optax.adam(config.critic_lr)
        
        self._critic_state = TrainStateWithTarget.create(
            apply_fn=critic.apply,
            params=critic_params,
            tx=critic_optimizer,
            target_params=target_critic_params
        )
        
        # initialize temperature
        temperature = Temperature(
            config.initial_temp, name="sac_temperature"
        )
        temp_params = temperature.init(temp_rng)["params"]
        temp_optimizer = optax.adam(config.temp_lr)
        
        self._temp_state = TrainState.create(
            apply_fn=temperature.apply,
            params=temp_params,
            tx=temp_optimizer
        )
        
        # set target temperature
        self._target_entropy = config.target_entropy or -config.action_dim / 2
        
    def act(self, observation: jax.Array, eval: bool) -> jax.Array:
        """SAC action function."""
        
        self._rng, act_rng = jax.random.split(self._rng)
        return act(self._actor_state, observation, eval, act_rng)
    
    def update(self, batch: Batch, step: int) -> MetricsDict:
        """SAC update step."""
        del step
        
        self._rng, critic_update_rng, actor_update_rng = jax.random.split(self._rng)
        
        # first update critic
        self._critic_state, critic_metrics = update_critic(
            self._critic_state, self._actor_state, self._temp_state,
            batch, self._config.gamma, self._config.tau, critic_update_rng
        )
        
        # then update actor
        self._actor_state, actor_metrics = update_actor(
            self._actor_state, self._critic_state, self._temp_state,
            batch, actor_update_rng
        )
        
        # next update temperature
        self._temp_state, temp_metrics = update_temperature(
            self._temp_state, actor_metrics["entropy"], self._target_entropy
        )
        
        return {**critic_metrics, **actor_metrics, **temp_metrics}