import flax
from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
import distrax
from typing import Sequence, Callable, Tuple, Union, Optional, NamedTuple
from ml_collections import ConfigDict
import functools

from jax_sandbox.nets import MLP, ResNetDownStack, NormedDense
from jax_sandbox.base_learner import Learner
from jax_sandbox.dataset import OnPolicyBatch, OnPolicyBatchWithLogits, make_minibatches
from jax_sandbox.utils import MetricsDict
from jax_sandbox.policy_gradient.utils import PGOutput, compute_gae


class PPGAuxOutput(NamedTuple):
    policy: distrax.Distribution
    values: jax.Array
    aux_values: Optional[jax.Array]


class IMPALAEncoder(nn.Module):
    """IMPALA encoder for Procgen envs."""
    
    channels: Sequence[int] = (16, 32, 32)
    use_normed_conv: bool = True
    use_normed_dense: bool = True
    use_batch_norm: bool = True
    norm_scale: float = 1.0
    activation: Callable[[jax.Array], jax.Array] = nn.relu
    w_init: Callable[[jax.Array], jax.Array] = nn.initializers.he_uniform
    use_max_pool: bool = True
    num_blocks: int = 2
    final_dense_scale: float = 1.4
    activate_final: bool = True
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = x / 255.0
        
        for i, ch in enumerate(self.channels):
            x = ResNetDownStack(
                ch, use_normed_conv=self.use_normed_conv,
                use_batch_norm=self.use_batch_norm, norm_scale=self.norm_scale,
                activation=self.activation, w_init=self.w_init,
                use_max_pool=self.use_max_pool, num_blocks=self.num_blocks,
                name=f"resblock_{i}"
            )(x)
            
        dense_cls = functools.partial(NormedDense, scale=self.final_dense_scale) if self.use_normed_dense else nn.Dense
        x = dense_cls(
            256, kernel_init=self.w_init,
        )(x)
        if self.activate_final:
            x = self.activation(x)
        
        return x
    
    
class ConvActorCritic(IMPALAEncoder):
    """Convolutional actor-critic."""
    
    num_actions: int
    head_scale: float = 0.1
    add_aux_head: bool = False
    
    @nn.compact
    def __call__(self, x: jax.Array) -> Tuple[PPGAuxOutput, PGOutput]:
        
        x = super().__call__(x)
        
        # now add actor + critic heads
        dense_cls = functools.partial(NormedDense, scale=self.head_scale) if self.use_normed_dense else nn.Dense
        logits = dense_cls(
            self.num_actions, kernel_init=self.w_init,
        )(x)
        dist = distrax.Categorical(logits=logits)
        
        value_inp = jnp.where(
            self.add_aux_head,
            jax.lax.stop_gradient(x),
            x
        )
        values = dense_cls(
            1, kernel_init=self.w_init
        )(value_inp)
        
        if self.add_aux_head:
            aux_values = dense_cls(
                1, kernel_init=self.w_init,
            )(x)
        else:
            aux_values = None
            
        return PPGAuxOutput(
            policy=dist, values=values, aux_values=aux_values
        )
        
        
class ConvCritic(IMPALAEncoder):
    """Convolutional critic only."""
    
    head_scale: float = 0.1
    
    @nn.compact
    def __call__(self, x: jax.Array) -> Tuple[PPGAuxOutput, PGOutput]:
        
        x = super().__call__(x)
        
        # now add actor + critic heads
        dense_cls = functools.partial(NormedDense, scale=self.head_scale) if self.use_normed_dense else nn.Dense
        values = dense_cls(
            1, kernel_init=self.w_init,
        )(x)
        return values
    
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
def ppo_minibatch_update_shared(
    actor_critic_state: TrainState,
    minibatch: OnPolicyBatch,
    gamma: float,
    gae_lam: float,
    vf_coef: float,
    ent_coef: float,
    clip_range: float,
    clip_vloss: bool,
) -> Tuple[TrainState, MetricsDict]:
    """Minibatch PPO update. Done in inner loop. Assumes we are in shared PPG regime, and policy outputs aux values."""
    
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


@functools.partial(jax.jit, static_argnames=("gamma", "gae_lam", "vf_coef", "ent_coef", "clip_range", "clip_vloss"))
def ppo_minibatch_update_separate(
    actor_critic_state: TrainState,
    critic_state: TrainState,
    minibatch: OnPolicyBatch,
    gamma: float,
    gae_lam: float,
    ent_coef: float,
    clip_range: float,
    clip_vloss: bool,
) -> Tuple[TrainState, TrainState, MetricsDict]:
    """Minibatch PPO update. Done in inner loop. Assumes we are in separate PPG regime, and we update both actor and critic."""
    
    # first, compute the final values so they're of shape [T + 1]
    next_values = critic_state.apply_fn(
        {"params": critic_state.params}, minibatch.next_observations
    )
    curr_values = critic_state.apply_fn(
        {"params": critic_state.params}, minibatch.observations
    )
    values = jnp.stack([curr_values, next_values[-1]], axis=0)
    values = jax.lax.stop_gradient(values)
    
    # now compute advantages
    advantages = compute_gae(
        minibatch.rewards, minibatch.not_dones, values, gamma, gae_lam
    )
    advantages = jax.lax.stop_gradient(advantages)
    
    # now do optimization step
    def ppo_actor_loss(params: flax.core.FrozenDict) -> Tuple[jax.Array, MetricsDict]:
        """PPO policy loss function."""
        
        new_dist = actor_critic_state.apply_fn(
            {"params": params}, minibatch.observations
        ).policy
        new_log_probs = new_dist.log_prob(minibatch.actions)
        
        # compute PG loss
        ratio = jnp.exp(new_log_probs - minibatch.log_probs)
        unclipped_pg_loss = -advantages * ratio
        clipped_pg_loss = -advantages * jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
        pg_loss = jnp.maximum(unclipped_pg_loss, clipped_pg_loss).mean()
        
        # entropy loss
        entropy = new_dist.entropy().mean()
        
        # total loss
        loss = pg_loss - ent_coef * entropy
        
        # compute metrics
        old_approx_kl = -(new_log_probs - minibatch.log_probs)
        new_approx_kl = (ratio - 1 - jnp.log(ratio)).mean()
        clipfrac = (jnp.abs(ratio - 1.0) > clip_range).astype(jnp.float32).mean()
        
        return loss, {
            "pg_loss": pg_loss,
            "entropy": entropy,
            "old_approx_kl": old_approx_kl,
            "new_approx_kl": new_approx_kl,
            "clip_frac": clipfrac,
        }
        
    actor_grads, actor_metrics = jax.grad(ppo_actor_loss, has_aux=True)(actor_critic_state.params)
    new_actor_critic_state = actor_critic_state.apply_gradients(grads=actor_grads)
    
    def ppo_critic_loss(params: flax.core.FrozenDict) -> Tuple[jax.Array, MetricsDict]:
        """PPO critic loss function."""
        
        new_values = critic_state.apply_fn(
            {"params": params}, minibatch.observations
        )
        
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
            
        # compute explained variance
        return_var = jnp.var(returns)
        explained_var = jnp.where(
            return_var == 0,
            jnp.nan,
            1.0 - jnp.var(values - returns) / return_var
        )
        
        return v_loss, {
            "explained_variance": explained_var
        }
        
    critic_grads, critic_metrics = jax.grad(ppo_critic_loss, has_aux=True)(critic_state.params)
    new_critic_state = critic_state.apply_gradients(grads=critic_grads)
    
    metrics = {**actor_metrics, **critic_metrics}
    return new_actor_critic_state, new_critic_state, metrics


@functools.partial(jax.jit, static_argnames=("bc_coef",))
def aux_update_shared(
    actor_critic_state: TrainState,
    minibatch: OnPolicyBatchWithLogits,
    bc_coef: float,
) -> Tuple[TrainState, MetricsDict]:
    """PPG auxiliary update, assuming separate actor and critic networks."""
    
    returns = minibatch.returns_to_go
    
    def aux_loss(params: flax.core.FrozenDict) -> Tuple[jax.Array, MetricsDict]:
        """Auxiliary loss function."""
        
        policy_output = actor_critic_state.apply_fn(
            {"params": params}, minibatch.observations
        )
        dist = policy_output.dist
        values = policy_output.values
        aux_values = policy_output.aux_values
        
        # KL loss
        old_dist = distrax.Categorical(logits=minibatch.logits)
        kl_div = old_dist.kl_divergence(dist).mean()
        
        # Auxiliary value function MSE
        aux_v_loss = 0.5 * ((aux_values - returns) ** 2).mean()
        
        # Optimize critic value function more
        real_v_loss = 0.5 * ((values - returns) ** 2).mean()
        
        # combine losses
        loss = aux_v_loss + real_v_loss + bc_coef * kl_div
        return loss, {
            "kl": kl_div,
            "aux_v_loss": aux_v_loss,
            "real_v_loss": real_v_loss
        }
        
    grads, metrics = jax.grad(aux_loss, has_aux=True)(actor_critic_state.params)
    new_actor_critic_state = actor_critic_state.apply_gradients(grads=grads)
    
    return new_actor_critic_state, metrics


@functools.partial(jax.jit, static_argnames=("bc_coef",))
def aux_update_separate(
    actor_critic_state: TrainState,
    critic_state: TrainState,
    minibatch: OnPolicyBatchWithLogits,
    bc_coef: float,
) -> Tuple[TrainState, TrainState, MetricsDict]:
    """Auxiliary loss, when policy and value are separate."""
    
    returns = minibatch.returns_to_go
    
    def actor_aux_loss(params: flax.core.FrozenDict) -> Tuple[jax.Array, MetricsDict]:
        """Actor auxiliary loss."""
        
        policy_output = actor_critic_state.apply_fn(
            {"params": params}, minibatch.observations
        )
        dist = policy_output.dist
        aux_values = policy_output.values
        
        # KL loss
        old_dist = distrax.Categorical(logits=minibatch.logits)
        kl_div = old_dist.kl_divergence(dist).mean()
        
        # Auxiliary value function MSE
        aux_v_loss = 0.5 * ((aux_values - returns) ** 2).mean()
        
        # combine the losses
        loss = aux_v_loss + bc_coef * kl_div
        return loss, {
            "kl": kl_div,
            "aux_v_loss": aux_v_loss
        }
        
    actor_grads, actor_metrics = jax.grad(actor_aux_loss, has_aux=True)(actor_critic_state.params)
    new_actor_critic_state = actor_critic_state.apply_gradients(grads=actor_grads)
    
    # optimize critic now
    def critic_aux_loss(params: flax.core.FrozenDict) -> Tuple[jax.Array, MetricsDict]:
        """Critic auxiliary loss."""
        
        values = critic_state.apply_fn(
            {"params": params}, minibatch.observations
        )
        real_v_loss = 0.5 * ((values - returns) ** 2).mean()
        
        return real_v_loss, {
            "real_v_loss": real_v_loss
        }
        
    critic_grads, critic_metrics = jax.grad(critic_aux_loss, has_aux=True)(critic_state.params)
    new_critic_state = critic_state.apply_gradients(grads=critic_grads)
    
    metrics = {**actor_metrics, **critic_metrics}
    return new_actor_critic_state, new_critic_state, metrics

# ========================= Agent construction. =========================

class PPG(Learner):
    """Phasic policy gradient."""
    
    def __init__(self, config: ConfigDict):
        self._config = config
        
        # init
        rng = jax.random.PRNGKey(config.seed)
        self._rng, init_rng = jax.random.split(rng)
        
        w_init = getattr(nn.initializers, config.w_init)
        activation = getattr(nn, config.activation)
        actor_critic = ConvActorCritic(
            self._config.channels, self._config.use_normed_conv, self._config.use_normed_dense,
            self._config.use_batch_norm, self._config.norm_scale, activation, w_init,
            self._config.use_max_pool, self._config.num_blocks, self._config.final_dense_scale,
            self._config.activate_before_heads, self._config.action_dim, self._config.head_scale,
            self._config.add_aux_head, name="ppg_actor_critic"
        )
        actor_critic_params = actor_critic.init(
            init_rng, jnp.zeros((1, config.observation_dim))
        )["params"]
        actor_critic_optimizer = optax.adam(config.lr)
        
        # if global norm is specified, we clip gradients before we run through adam
        if self._config.max_grad_norm is not None:
            actor_critic_optimizer = optax.chain(
                optax.clip_by_global_norm(self._config.max_grad_norm),
                actor_critic_optimizer
            )
            
        self._actor_critic_state = TrainState.create(
            apply_fn=actor_critic.apply,
            params=actor_critic_params,
            tx=actor_critic_optimizer
        )
            
        # initialize critic if needed
        if not self._config.add_aux_head:
            self._rng, critic_init_rng = jax.random.split(self._rng)
            critic = ConvCritic(
                self._config.channels, self._config.use_normed_conv, self._config.use_normed_dense,
                self._config.use_batch_norm, self._config.norm_scale, activation, w_init,
                self._config.use_max_pool, self._config.num_blocks, self._config.final_dense_scale,
                self._config.activate_before_heads, self._config.head_scale, name="ppg_sep_critic"
            )
            critic_params = critic.init(
                critic_init_rng, jnp.zeros((1, config.observation_dim))
            )["params"]
            critic_optimizer = optax.adam(config.lr)
            
            if self._config.max_grad_norm is not None:
                critic_optimizer = optax.chain(
                    optax.clip_by_global_norm(self._config.max_grad_norm),
                    critic_optimizer
                )
            
            self._critic_state = TrainState.create(
                apply_fn=critic.apply,
                params=critic_params,
                tx=critic_optimizer
            )
        
    def act(self, observation: jax.Array, eval: jax.Array) -> jax.Array:
        self._rng, act_rng = jax.random.split(self._rng)
        return act(self._actor_critic_state, observation, eval, act_rng)
    
    def update(self, batch: OnPolicyBatch, step: int) -> MetricsDict:
        """Standard PPO update here. Auxiliary update is on an off-policy batch."""
        
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
        
        update_fn = (
            functools.partial(ppo_minibatch_update_shared, vf_coef=self._config.vf_coef) 
            if self._config.add_aux_head
            else functools.partial(ppo_minibatch_update_separate, critic_state=self._critic_state)
        )
        def update_loop_body(state: TrainState, minibatch: OnPolicyBatch) -> Tuple[TrainState, MetricsDict]:
            new_state, metrics = update_fn(
                state, minibatch, self._config.gamma, self._config.gae_lam,
                self._config.ent_coef, self._config.clip_range, self._config.clip_vloss
            )
            return new_state, metrics
        
        self._actor_critic_state, metrics = jax.lax.scan(
            update_loop_body, init=self._actor_critic_state, xs=make_minibatches(batch, self._config.minibatch_size)
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)

        return metrics
    
    