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
from jax_sandbox.policy_gradient.utils import PGOutput, compute_gae


class ActorCritic(nn.Module):
    """Actor-critic module. Same as in REINFORCE."""
    
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
                "logstd", nn.initializers.zeros, mean.shape
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
    """Jitted function on how to act. Stolen from REINFORCE code -- it's basically the same function."""
    
    dist = actor_critic_state.apply_fn(
        {"params": actor_critic_state.params}, observation
    ).policy
    return jnp.where(
        eval,
        dist.mode(),
        dist.sample(seed=rng),
    )
    

@functools.partial(jax.jit, static_argnames=("gamma", "gae_lam", "vf_coef", "ent_coef", "clip_range", "clip_vloss"))
def minibatch_update(
    actor_critic_state: TrainState,
    minibatch: OnPolicyBatch,
    gamma: float,
    gae_lam: float,
    vf_coef: float,
    ent_coef: float,
    clip_range: float,
    clip_vloss: bool,
) -> Tuple[TrainState, MetricsDict]:
    """Minibatch PPO update."""
    
    # first, compute the final values so they're of shape [T + 1]
    next_values = actor_critic_state.apply_fn(
        {"params": actor_critic_state.params}, minibatch.next_observations
    ).values
    curr_values = actor_critic_state.apply_fn(
        {"params": actor_critic_state.params}, minibatch.observations
    ).values
    values = jnp.stack([curr_values, next_values[-1]], axis=0)
    values = jax.lax.stop_gradient(values)
    
    # now compute advantages
    advantages = compute_gae(
        minibatch.rewards, minibatch.not_dones, values, gamma, gae_lam
    )
    advantages = jax.lax.stop_gradient(advantages)
    
    # now do optimization step
    def ppo_loss(params: flax.core.FrozenDict) -> Tuple[jax.Array, MetricsDict]:
        """PPO loss function."""
        
        new_output = actor_critic_state.apply_fn(
            {"params": params}, minibatch.observations
        )
        new_dist, new_values = new_output.policy, new_output.values
        new_log_probs = new_dist.log_prob(minibatch.actions)
        
        # compute PG loss
        ratio = jnp.exp(new_log_probs - minibatch.log_probs)
        unclipped_pg_loss = -advantages * ratio
        clipped_pg_loss = -advantages * jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
        pg_loss = jnp.maximum(unclipped_pg_loss, clipped_pg_loss).mean()
        
        # compute value loss
        returns = advantages + values
        if clip_vloss:
            v_loss_unclipped = (new_values - returns) ** 2
            
            new_clipped_values = values + jnp.clip(
                new_values - values, -clip_range, clip_range
            )
            v_loss_clipped = (new_clipped_values - returns) ** 2
            
            v_loss = jnp.maximum(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss.mean()
        else:
            v_loss = 0.5 * ((new_values - returns) ** 2).mean()
            
        # entropy loss
        entropy = new_dist.entropy().mean()
        
        # total loss
        loss = pg_loss + vf_coef * v_loss - ent_coef * entropy
        
        # compute metrics
        old_approx_kl = -(new_log_probs - minibatch.log_probs)
        new_approx_kl = (ratio - 1 - jnp.log(ratio)).mean()
        clipfrac = (jnp.abs(ratio - 1.0) > clip_range).astype(jnp.float32).mean()
        
        return_var = jnp.var(returns)
        explained_var = jnp.where(
            return_var == 0,
            jnp.nan,
            1.0 - jnp.var(values - returns) / return_var
        )
        
        return loss, {
            "pg_loss": pg_loss,
            "value_loss": v_loss,
            "entropy": entropy,
            "old_approx_kl": old_approx_kl,
            "new_approx_kl": new_approx_kl,
            "clip_frac": clipfrac,
            "explained_variance": explained_var
        }
        
    grads, metrics = jax.grad(ppo_loss, has_aux=True)(actor_critic_state.params)
    new_actor_critic_state = actor_critic_state.apply_gradients(grads=grads)
    
    return new_actor_critic_state, metrics

# ========================= Agent construction. =========================

class PPO(Learner):
    """Proximal policy optimization."""
    
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
        
        # if global norm is specified, we clip gradients before we run through adam
        if self._config.max_grad_norm is not None:
            actor_critic_optimizer = optax.chain(
                optax.clip_by_global_norm(self._config.max_grad_norm),
                actor_critic_optimizer
            )
        
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
            new_state, metrics = minibatch_update(
                state, minibatch, self._config.gamma, self._config.gae_lam,
                self._config.vf_coef, self._config.ent_coef,
                self._config.clip_range, self._config.clip_vloss
            )
            return new_state, metrics
        
        self._state, metrics = jax.lax.scan(
            update_loop_body, init=self._state, xs=make_minibatches(batch, self._config.minibatch_size)
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        
        return metrics