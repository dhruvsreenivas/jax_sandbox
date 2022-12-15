import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import NamedTuple, Optional, Tuple, Dict

from common.nets import *
from common.dataset import TransitionBatch
from common.utils import opt_class

class SACState(NamedTuple):
    actor_params: hk.Params
    critic_params: hk.Params
    target_critic_params: hk.Params
    log_alpha_params: Optional[hk.Params]
    
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    log_alpha_opt_state: Optional[optax.OptState]
    
    rng_key: jax.random.PRNGKey

class SAC:
    def __init__(self, cfg):
        # set up actor + critic
        if cfg.img_input:
            channels = list(cfg.channels)
            kernels = list(cfg.kernels)
            strides = list(cfg.strides)
            
        assert cfg.continuous, "SAC is for continuous control envs."
        assert not cfg.deterministic, "SAC requires entropy term, which means we need action distribution."
        
        if cfg.img_input:
            def actor_fn(s):
                net = hk.Sequential([
                    ConvTorso(channels, kernels, strides, act=cfg.activation, activate_final=cfg.activate_final),
                    LinearGaussian(cfg.action_shape, softplus=cfg.softplus, min_std=cfg.min_std)
                ])
                return net(s)

            def critic_fn(s, a):
                sa = jnp.concatenate([s, a], axis=-1)
                
                sa_rep = ConvTorso(channels, kernels, strides, act=cfg.activation, activate_final=cfg.activate_final)(sa)
                q1, q2 = DoubleLinear(1)(sa_rep)
                return q1, q2
            
        else:
            def actor_fn(s):
                net = hk.Sequential([
                    MLP(cfg.hidden_sizes, act=cfg.activation, activate_final=cfg.activate_final, use_ln=cfg.use_ln),
                    LinearGaussian(cfg.action_shape, softplus=cfg.softplus, min_std=cfg.min_std)
                ])
                return net(s)
            
            def critic_fn(s, a):
                sa = jnp.concatenate([s, a], dim=-1)
                
                sa_rep = MLP(cfg.hidden_sizes, act=cfg.activation, activate_final=cfg.activate_final, use_ln=cfg.use_ln)(sa)
                q1, q2 = DoubleLinear(1)(sa_rep)
                return q1, q2
            
        actor = hk.without_apply_rng(hk.transform(actor_fn))
        critic = hk.without_apply_rng(hk.transform(critic_fn))
        
        # init
        key = jax.random.PRNGKey(cfg.seed)
        actor_key, critic_key, state_key = jax.random.split(key, 3)
        
        actor_params = actor.init(actor_key, batched_zeros_like(cfg.obs_shape))
        critic_params = target_critic_params = critic.init(critic_key, batched_zeros_like(cfg.obs_shape), batched_zeros_like(cfg.action_shape))
        
        actor_opt = opt_class(cfg.optim)(learning_rate=cfg.actor_lr)
        actor_opt_state = actor_opt.init(actor_params)
        critic_opt = opt_class(cfg.optim)(learning_rate=cfg.critic_lr)
        critic_opt_state = critic_opt.init(critic_params)
        
        if cfg.tune_alpha:
            log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
            alpha_opt = opt_class(cfg.optim)(learning_rate=cfg.alpha_lr)
            alpha_opt_state = alpha_opt.init(log_alpha)
        else:
            log_alpha = None
            alpha_opt_state = None
            
        self._state = SACState(
            actor_params=actor_params,
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            log_alpha_params=log_alpha,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            alpha_opt_state=alpha_opt_state,
            rng_key=state_key
        )
        
        # hparams
        gamma = cfg.gamma
        tau = cfg.tau
        target_entropy = cfg.target_entropy
        
        # functions
        def act(state: jnp.ndarray, eval_mode: bool) -> jnp.ndarray:
            action_dist = actor.apply(self._state.actor_params, state)
            
            if eval_mode:
                action = action_dist.mode()
            else:
                key, subkey = jax.random.split(self._state.rng_key)
                action = action_dist.sample(seed=subkey)
                
                self._state = self._state._replace(rng_key=key)
                
            return action
        
        @jax.jit
        def critic_loss_fn(critic_params: hk.Params,
                           target_critic_params: hk.Params,
                           actor_params: hk.Params,
                           log_alpha: Optional[hk.Params],
                           key: jax.random.PRNGKey,
                           batch: TransitionBatch) -> jnp.ndarray:
            
            # get targets
            next_action_dist = actor.apply(actor_params, batch.next_states)
            next_actions = next_action_dist.sample(seed=key)
            next_lps = next_action_dist.log_prob(next_actions).sum(-1)
            
            next_q1, next_q2 = critic.apply(target_critic_params, batch.next_states, next_actions)
            next_q = jnp.minimum(next_q1, next_q2)
            if cfg.tune_alpha:
                alpha = jnp.exp(log_alpha)
            else:
                alpha = cfg.alpha
            
            next_q = next_q - alpha * next_lps
            next_v = batch.rewards + gamma * (1.0 - batch.dones) * jnp.squeeze(next_q)
            next_v = jax.lax.stop_gradient(next_v)
            
            # current estimates
            curr_q1, curr_q2 = critic.apply(critic_params, batch.states, batch.actions)
            curr_q1 = jnp.squeeze(curr_q1)
            curr_q2 = jnp.squeeze(curr_q2)
            
            loss = jnp.mean(jnp.square(curr_q1 - next_v)) + jnp.mean(jnp.square(curr_q2, next_v))
            return loss
        
        @jax.jit
        def actor_loss_fn(actor_params: hk.Params,
                          critic_params: hk.Params,
                          log_alpha: Optional[hk.Params],
                          key: jax.random.PRNGKey,
                          batch: TransitionBatch) -> jnp.ndarray:
            
            action_dist = actor.apply(actor_params, batch.states)
            actions = action_dist.sample(seed=key)
            lps = action_dist.log_prob(actions).sum(-1)
            
            q1, q2 = critic.apply(critic_params, batch.states, actions)
            if cfg.tune_alpha:
                alpha = jnp.exp(log_alpha)
            else:
                alpha = cfg.alpha
            
            q = jnp.minimum(q1, q2).squeeze() - alpha * lps
            loss = -jnp.mean(q)
            return loss
        
        @jax.jit
        def alpha_loss_fn(log_alpha: hk.Params,
                          actor_params: hk.Params,
                          key: jax.random.PRNGKey,
                          batch: TransitionBatch) -> jnp.ndarray:
            
            action_dist = actor.apply(actor_params, batch.states)
            action = action_dist.sample(seed=key)
            lps = action_dist.log_prob(action).sum(-1)
            alpha = jnp.exp(log_alpha)
            
            alpha_loss = alpha * jax.lax.stop_gradient(-lps - target_entropy)
            return jnp.mean(alpha_loss)
        
        def update(state: SACState, batch: TransitionBatch, step: int) -> Tuple[SACState, Dict]:
            del step
            
            # keys
            critic_key, actor_key, alpha_key, state_key = jax.random.split(state.rng_key, 4)
            
            # update critic
            critic_loss_grad_fn = jax.value_and_grad(critic_loss_fn)
            critic_loss, critic_grads = critic_loss_grad_fn(state.critic_params,
                                                           state.target_critic_params,
                                                           state.actor_params,
                                                           state.log_alpha_params,
                                                           critic_key,
                                                           batch)
            
            critic_update, new_critic_opt_state = critic_opt.update(critic_grads, state.critic_opt_state)
            new_critic_params = optax.apply_updates(state.critic_params, critic_update)
            
            # update actor (and alpha)
            actor_loss_grad_fn = jax.value_and_grad(actor_loss_fn)
            actor_loss, actor_grads = actor_loss_grad_fn(state.actor_params,
                                                         new_critic_params,
                                                         state.log_alpha_params,
                                                         actor_key,
                                                         batch)
            
            actor_update, new_actor_opt_state = actor_opt.update(actor_grads, state.actor_opt_state)
            new_actor_params = optax.apply_updates(state.actor_params, actor_update)
            
            if cfg.tune_alpha:
                alpha_loss_grad_fn = jax.value_and_grad(alpha_loss_fn)
                alpha_loss, alpha_grads = alpha_loss_grad_fn(state.log_alpha_params,
                                                             new_actor_params,
                                                             alpha_key,
                                                             batch)
                
                alpha_update, new_alpha_opt_state = alpha_opt.update(alpha_grads, state.log_alpha_opt_state)
                new_alpha_params = optax.apply_updates(state.log_alpha_params, alpha_update)
            else:
                alpha_loss = 0.0
                new_alpha_params = None
                new_alpha_opt_state = None
                
            new_target_critic_params = update_target(new_critic_params, state.target_critic_params, tau)
            
            state = SACState(
                actor_params=new_actor_params,
                critic_params=new_critic_params,
                target_critic_params=new_target_critic_params,
                log_alpha_params=new_alpha_params,
                actor_opt_state=new_actor_opt_state,
                critic_opt_state=new_critic_opt_state,
                log_alpha_opt_state=new_alpha_opt_state,
                rng_key=state_key
            )
            
            metrics = {
                'actor_loss': actor_loss,
                'critic_loss': critic_loss,
                'alpha_loss': alpha_loss
            }
            
            return state, metrics
        
        self._act = jax.jit(act)
        self._update = jax.jit(update)