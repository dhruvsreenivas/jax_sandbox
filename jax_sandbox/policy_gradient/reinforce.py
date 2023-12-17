import flax
from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
import distrax
from typing import Sequence, Callable, Tuple
from ml_collections import ConfigDict
import functools

from jax_sandbox.nets import MLP
from jax_sandbox.base_learner import Learner
from jax_sandbox.dataset import OnPolicyBatch, make_minibatches
from jax_sandbox.utils import MetricsDict
from jax_sandbox.policy_gradient.utils import PGOutput


class ActorCritic(nn.Module):
    """Actor-critic architecture."""
    
    action_dim: int
    hidden_sizes: Sequence[int]
    w_init: Callable[[jax.Array], jax.Array] = nn.initializers.he_uniform
    activation: Callable[[jax.Array], jax.Array] = nn.relu
    discrete: bool
    
    @nn.compact
    def __call__(self, x: jax.Array) -> PGOutput:
        
        policy_dist_params = MLP(
            hidden_sizes=self.hidden_sizes,
            output_size=self.action_dim,
            w_init=self.w_init,
            activation=self.activation,
            name="pi"
        )(x)
        
        values = MLP(
            hidden_sizes=self.hidden_sizes,
            output_size=1,
            w_init=self.w_init,
            activation=self.activation,
            name="v"
        )(x)
        
        if self.discrete:
            dist = distrax.Categorical(logits=policy_dist_params)
        else:
            mean = policy_dist_params
            logstd = self.param(
                "logstd", nn.initializers.constant(-0.5), mean.shape
            )
            dist = distrax.MultivariateNormalDiag(
                loc=mean, scale=jnp.exp(logstd)
            )
            
        return PGOutput(
            policy=dist, values=values
        )
    
# ========================= Action + loss functions and update steps =========================

@jax.jit
def act(
    actor_critic_state: TrainState, observation: jax.Array, eval: bool, rng: jax.random.PRNGKey
) -> jax.Array:
    """Jitted function on how to act."""
    
    dist = actor_critic_state.apply_fn(
        {"params": actor_critic_state.params}, observation
    ).policy
    return jnp.where(
        eval,
        dist.mode(),
        dist.sample(seed=rng),
    )

@functools.partial(jax.jit, static_argnames=("vf_coef",))
def minibatch_update(
    actor_critic_state: TrainState,
    minibatch: OnPolicyBatch,
    vf_coef: float,
) -> Tuple[TrainState, MetricsDict]:
    """Minibatch actor-critic update."""
    
    # here we assume the batch is a batch of trajectories, in order:
    # i.e. "observations" -> [B, T, observation_dims].
    
    def reinforce_loss(params: flax.core.FrozenDict) -> Tuple[jax.Array, MetricsDict]:
        """Policy gradient loss."""
        
        output = actor_critic_state.apply_fn(
            {"params": params}, minibatch.observations
        )
        dist, values = output.policy, output.values
        log_probs = dist.log_prob(minibatch.actions) # [B, T]
        
        # PG loss: max E[log pi(a | s) * (returns_to_go - values)]
        advantages = minibatch.returns_to_go - values
        pg_loss = -(log_probs * advantages).mean()
        
        # Value loss: min E[(V(s) - returns_to_go) ** 2]
        value_loss = jnp.mean(jnp.square(values - minibatch.returns_to_go))
        
        return pg_loss + vf_coef * value_loss, {"pg_loss": pg_loss, "value_loss": value_loss}
    
    grads, metrics = jax.grad(reinforce_loss, has_aux=True)(actor_critic_state.params)
    new_actor_critic_state = actor_critic_state.apply_gradients(grads=grads)
    
    return new_actor_critic_state, metrics

# ========================= Agent construction. =========================

class REINFORCE(Learner):
    """REINFORCE algorithm, with added value baseline."""
    
    def __init__(self, config: ConfigDict):
        self._config = config
        
        # init
        rng = jax.random.PRNGKey(config.seed)
        self._rng, init_rng = jax.random.split(rng)
        
        w_init = getattr(nn.initializers, config.w_init)
        activation = getattr(nn, config.activation)
        actor_critic = ActorCritic(
            config.action_dim, config.hidden_sizes, w_init=w_init,
            activation=activation, discrete=config.discrete, name="actor_critic"
        )
        actor_critic_params = actor_critic.init(
            init_rng, jnp.zeros((1, config.observation_dim))
        )
        actor_critic_optimizer = optax.adam(config.lr)
        
        self._state = TrainState.create(
            apply_fn=actor_critic.apply,
            params=actor_critic_params,
            tx=actor_critic_optimizer
        )
        
    def act(self, observation: jax.Array, eval: jax.Array) -> jax.Array:
        self._rng, act_rng = jax.random.split(self._rng)
        return act(self._state, observation, eval, act_rng)
    
    def update(self, batch: OnPolicyBatch, step: int) -> MetricsDict:
        """Full batch update."""
        del step
        
        bs = batch.observations.shape[0]
        assert bs % self._config.minibatch_size == 0, "Batch size must be divisible by minibatch size."
        
        # first shuffle and create minibatches
        self._rng, perm_rng = jax.random.split(self._rng)
        
        perm = jax.random.permutation(perm_rng, bs)
        batch.observations = batch.observations[perm]
        batch.actions = batch.actions[perm]
        batch.rewards = batch.rewards[perm]
        batch.next_observations = batch.next_observations[perm]
        batch.not_dones = batch.not_dones[perm]
        batch.log_probs = batch.log_probs[perm]
        batch.returns_to_go = batch.returns_to_go[perm]
        
        def update_loop_body(state: TrainState, minibatch: OnPolicyBatch) -> Tuple[TrainState, MetricsDict]:
            new_state, metrics = minibatch_update(state, minibatch, self._config.vf_coef)
            return new_state, metrics
        
        self._state, metrics = jax.lax.scan(
            update_loop_body, init=self._state, xs=make_minibatches(batch, self._config.minibatch_size)
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        
        return metrics