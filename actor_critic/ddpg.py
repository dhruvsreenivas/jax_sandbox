import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import NamedTuple, Tuple, Dict

from common.nets import *
from common.dataset import TransitionBatch
from common.utils import opt_class

class DDPGState(NamedTuple):
    actor_params: hk.Params
    critic_params: hk.Params
    target_actor_params: hk.Params
    target_critic_params: hk.Params
    
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    
    rng_key: jax.random.PRNGKey

class DDPG:
    def __init__(self, cfg):
        if cfg.img_input:
            channels = list(cfg.channels)
            kernels = list(cfg.kernels)
            strides = list(cfg.strides)
            
        assert cfg.continuous, "DDPG is for continuous control envs."
        
        if cfg.img_input:
            if cfg.deterministic:
                def actor_fn(s):
                    net = hk.Sequential([
                        ConvTorso(channels, kernels, strides, act=cfg.activation, activate_final=cfg.activate_final),
                        hk.Linear(cfg.action_shape)
                    ])
                    return net(s)
            else:
                def actor_fn(s):
                    net = hk.Sequential([
                        ConvTorso(channels, kernels, strides, act=cfg.activation, activate_final=cfg.activate_final),
                        LinearGaussian(cfg.action_shape, softplus=cfg.softplus, min_std=cfg.min_std)
                    ])
                    return net(s)
            
            def critic_fn(s, a):
                net = hk.Sequential([
                    ConvTorso(channels, kernels, strides, act=cfg.activation, activate_final=cfg.activate_final),
                    hk.Linear(1)
                ])
                
                sa = jnp.concatenate([s, a], -1)
                return net(sa)
        else:
            if cfg.deterministic:
                def actor_fn(s):
                    net = hk.Sequential([
                        MLP(cfg.hidden_sizes, act=cfg.activation, activate_final=cfg.activate_final, use_ln=cfg.use_ln),
                        hk.Linear(cfg.action_shape)
                    ])
                    return net(s)
            else:
                def actor_fn(s):
                    net = hk.Sequential([
                        MLP(cfg.hidden_sizes, act=cfg.activation, activate_final=cfg.activate_final, use_ln=cfg.use_ln),
                        LinearGaussian(cfg.action_shape, softplus=cfg.softplus, min_std=cfg.min_std)
                    ])
                    return net(s)
            
            def critic_fn(s, a):
                net = hk.Sequential([
                    MLP(cfg.hidden_sizes, act=cfg.activation, activate_final=cfg.activate_final, use_ln=cfg.use_ln),
                    hk.Linear(1)
                ])
                
                sa = jnp.concatenate([s, a], -1)
                return net(sa)
            
        actor = hk.without_apply_rng(hk.transform(actor_fn))
        critic = hk.without_apply_rng(hk.transform(critic_fn))
        
        # params
        key = jax.random.PRNGKey(cfg.seed)
        actor_key, critic_key, state_key = jax.random.split(key, 3)
        actor_params = target_actor_params = actor.init(actor_key, batched_zeros_like(cfg.obs_shape))
        critic_params = target_critic_params = critic.init(critic_key, batched_zeros_like(cfg.obs_shape), batched_zeros_like(cfg.action_shape))
        
        actor_opt = opt_class(cfg.optim)(learning_rate=cfg.actor_lr)
        critic_opt = opt_class(cfg.optim)(learning_rate=cfg.critic_lr)
        actor_opt_state = actor_opt.init(actor_params)
        critic_opt_state = critic_opt.init(critic_params)
        
        # set up state
        self._state = DDPGState(
            actor_params=actor_params,
            critic_params=critic_params,
            target_actor_params=target_actor_params,
            target_critic_params=target_critic_params,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            rng_key=state_key
        )
        
        # set up other hparams
        gamma = cfg.gamma
        tau = cfg.tau
        min_action = cfg.min_action
        max_action = cfg.max_action
        policy_noise = cfg.policy_noise
        
        # functions
        def act(state: jnp.ndarray, eval_mode: bool) -> jnp.ndarray:
            if cfg.deterministic:
                action = actor.apply(self._state.actor_params, state)
                
                key, sample_key = jax.random.split(self._state.rng_key)
                noise = policy_noise * jax.random.normal(seed=sample_key, shape=action.shape)
                action = jnp.clip(action + noise, min_action, max_action)
                
                self._state = self._state._replace(rng_key=key)
            else:
                action_dist = actor.apply(self._state.actor_params, state)
            
                if eval_mode:
                    action = action_dist.mean()
                else:
                    key, sample_key = jax.random.split(self._state.rng_key)
                    action = action_dist.sample(seed=sample_key)
                    
                    self._state = self._state._replace(rng_key=key)
            
            return action
        
        @jax.jit
        def critic_loss_fn(critic_params: hk.Params,
                           target_critic_params: hk.Params,
                           target_actor_params: hk.Params,
                           batch: TransitionBatch) -> jnp.ndarray:
            
            # compute targets
            if cfg.deterministic:
                next_actions = actor.apply(target_actor_params, batch.next_states)
            else:
                next_action_dist = actor.apply(target_actor_params, batch.next_states)
                next_actions = next_action_dist.mean()
            
            next_q = critic.apply(target_critic_params, batch.next_states, next_actions)
            next_v = batch.rewards + gamma * (1.0 - batch.dones) * jnp.squeeze(next_q)
            next_v = jax.lax.stop_gradient(next_v)
            
            # current estimates
            curr_q = critic.apply(critic_params, batch.states, batch.actions)
            curr_q = jnp.squeeze(curr_q)
            loss = jnp.mean(jnp.square(curr_q - next_v))
            
            return loss
        
        @jax.jit
        def actor_loss_fn(actor_params: hk.Params,
                          critic_params: hk.Params,
                          batch: TransitionBatch) -> jnp.ndarray:
            
            if cfg.deterministic:
                actions = actor.apply(actor_params, batch.states)
            else:
                action_dist = actor.apply(actor_params, batch.states)
                actions = action_dist.mean()
            
            q = critic.apply(critic_params, batch.states, actions)
            loss = -jnp.mean(q)
            return loss
        
        def update(state: DDPGState, batch: TransitionBatch, step: int) -> Tuple[DDPGState, Dict]:
            del step
            
            # update critic first
            critic_loss_grad_fn = jax.value_and_grad(critic_loss_fn)
            critic_loss, critic_grads = critic_loss_grad_fn(state.critic_params,
                                                            state.target_critic_params,
                                                            state.target_actor_params,
                                                            batch)
            
            critic_update, new_critic_opt_state = critic_opt.update(critic_grads, state.critic_opt_state)
            new_critic_params = optax.apply_updates(state.critic_params, critic_update)
            
            # update actor
            actor_loss_grad_fn = jax.value_and_grad(actor_loss_fn)
            actor_loss, actor_grads = actor_loss_grad_fn(state.actor_params,
                                                         new_critic_params,
                                                         batch)
            
            actor_update, new_actor_opt_state = actor_opt.update(actor_grads, state.actor_opt_state)
            new_actor_params = optax.apply_updates(state.actor_params, actor_update)
            
            # update targets
            new_target_actor_params = update_target(new_actor_params, state.target_actor_params, tau)
            new_target_critic_params = update_target(new_critic_params, state.target_critic_params, tau)
            
            # returning stuff
            new_state = DDPGState(
                actor_params=new_actor_params,
                critic_params=new_critic_params,
                target_actor_params=new_target_actor_params,
                target_critic_params=new_target_critic_params,
                actor_opt_state=new_actor_opt_state,
                critic_opt_state=new_critic_opt_state,
                rng_key=state.rng_key
            )
            
            metrics = {
                'actor_loss': actor_loss,
                'critic_loss': critic_loss
            }
            
            return new_state, metrics
        
        self._act = jax.jit(act)
        self._update = jax.jit(update)