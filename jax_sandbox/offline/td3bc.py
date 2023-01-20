import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import NamedTuple, Tuple, Dict

from jax_sandbox.common.nets import *
from jax_sandbox.common.dataset import TransitionBatch
from jax_sandbox.common.utils import opt_class

class TD3BCState(NamedTuple):
    actor_params: hk.Params
    target_actor_params: hk.Params
    critic_params: hk.Params
    target_critic_params: hk.Params
    
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    
    rng_key: jax.random.PRNGKey

class TD3BC:
    def __init__(self, cfg):
        # set up nets
        if cfg.img_input:
            channels = list(cfg.channels)
            kernels = list(cfg.kernels)
            strides = list(cfg.strides)
            
        assert cfg.continuous, "TD3 is for continuous control envs."
        
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
                        LinearGaussian(cfg.action_shape)
                    ])
                    return net(s)
                
            def critic_fn(s, a):
                sa = jnp.concatenate([s, a], axis=-1)
                sa_rep = ConvTorso(channels, kernels, strides, act=cfg.activation, activate_final=cfg.activate_final)(sa)
                q1 = hk.Linear(1)(sa_rep)
                q2 = hk.Linear(1)(sa_rep)
                
                return q1, q2
        else:
            if cfg.deterministic:
                def actor_fn(s):
                    net = hk.Sequential([
                        MLP(cfg.hidden_sizes, act=cfg.activation, activate_final=cfg.activate_final, output_act=cfg.output_act, use_ln=cfg.use_ln),
                        hk.Linear(cfg.action_shape)
                    ])
                    return net(s)
            else:
                def actor_fn(s):
                    net = hk.Sequential([
                        MLP(cfg.hidden_sizes, act=cfg.activation, activate_final=cfg.activate_final, output_act=cfg.output_act, use_ln=cfg.use_ln),
                        LinearGaussian(cfg.action_shape)
                    ])
                    return net(s)
                
            def critic_fn(s, a):
                sa = jnp.concatenate([s, a], axis=-1)
                sa_rep = MLP(cfg.hidden_sizes, act=cfg.activation, activate_final=cfg.activate_final, output_act=cfg.output_act, use_ln=cfg.use_ln)(sa)
                q1 = hk.Linear(1)(sa_rep)
                q2 = hk.Linear(1)(sa_rep)
                
                return q1, q2
            
        actor = hk.without_apply_rng(hk.transform(actor_fn))
        critic = hk.without_apply_rng(hk.transform(critic_fn))
        
        # init
        key = jax.random.PRNGKey(cfg.seed)
        actor_key, critic_key, state_key = jax.random.split(key, 3)
        actor_params = target_actor_params = actor.init(actor_key, batched_zeros_like(cfg.obs_shape))
        critic_params = target_critic_params = critic.init(critic_key, batched_zeros_like(cfg.obs_shape), batched_zeros_like(cfg.action_shape))
        
        actor_opt = opt_class(cfg.optim)(learning_rate=cfg.actor_lr)
        actor_opt_state = actor_opt.init(actor_params)
        
        critic_opt = opt_class(cfg.optim)(learning_rate=cfg.critic_lr)
        critic_opt_state = critic_opt.init(critic_params)
        
        self._state = TD3BCState(
            actor_params=actor_params,
            target_actor_params=target_actor_params,
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            rng_key=state_key
        )
        
        # hparams
        gamma = cfg.gamma
        tau = cfg.tau
        lmbda = cfg.bc_lmbda
        policy_noise = cfg.policy_noise
        noise_clip = cfg.noise_clip
        min_action = cfg.min_action
        max_action = cfg.max_action
        target_update_freq = cfg.target_update_freq
        
        # functions
        def act(state: jnp.ndarray, eval_mode: bool) -> jnp.ndarray:
            if cfg.deterministic:
                action = actor.apply(self._state.actor_params, state)
                key, subkey = jax.random.split(self._state.rng_key)
                noise = policy_noise * jax.random.normal(subkey, shape=action.shape)
                noise = jnp.clip(noise, -noise_clip, noise_clip)
                action = jnp.clip(action + noise, min_action, max_action)
            
                self._state = self._state._replace(rng_key=key)
            else:
                action_dist = actor.apply(self._state.actor_params, state)
                if eval_mode:
                    action = action_dist.mean()
                else:
                    key, subkey = jax.random.split(self._state.rng_key)
                    action = action_dist.sample(seed=subkey)
                    
                    self._state = self._state._replace(rng_key=key)
                
            return action
        
        @jax.jit
        def critic_loss_fn(critic_params: hk.Params,
                           target_critic_params: hk.Params,
                           target_actor_params: hk.Params,
                           key: jax.random.PRNGKey,
                           batch: TransitionBatch) -> jnp.ndarray:
            
            if cfg.deterministic:
                next_action = actor.apply(target_actor_params, batch.next_states)
                noise = policy_noise * jax.random.normal(key, shape=next_action.shape)
                noise = jnp.clip(noise, -noise_clip, noise_clip)
                next_action = jnp.clip(next_action + noise, min_action, max_action)
            else:
                next_action_dist = actor.apply(target_actor_params, batch.next_states)
                next_action = next_action_dist.sample(seed=key)
                
            next_q1, next_q2 = critic.apply(target_critic_params, batch.next_states, next_action)
            next_q = jnp.minimum(next_q1, next_q2).squeeze()
            next_v = batch.rewards + gamma * (1.0 - batch.dones) * next_q
            
            curr_q1, curr_q2 = critic.apply(critic_params, batch.states, batch.actions)
            curr_q1 = jnp.squeeze(curr_q1)
            curr_q2 = jnp.squeeze(curr_q2)
            
            loss = jnp.mean(jnp.square(curr_q1, next_v)) + jnp.mean(jnp.square(curr_q2, next_v))
            return loss
        
        @jax.jit
        def actor_loss_fn(actor_params: hk.Params,
                          critic_params: hk.Params,
                          batch: TransitionBatch):
            
            if cfg.deterministic:
                actions = actor.apply(actor_params, batch.states)
            else:
                action_dist = actor.apply(actor_params, batch.states)
                actions = action_dist.mean()
            
            q1, _ = critic.apply(critic_params, batch.states, actions)
            q_norm = jax.lax.stop_gradient(jnp.mean(jnp.abs(q1)))
            
            td3_loss = -lmbda / q_norm * jnp.mean(q1)
            bc_loss = jnp.mean(jnp.square(actions - batch.actions))
            return td3_loss + bc_loss
        
        def update(state: TD3BCState, batch: TransitionBatch, step: int) -> Tuple[TD3BCState, Dict]:
            critic_key, state_key = jax.random.split(state.rng_key)
            
            # update critic
            critic_loss_grad_fn = jax.value_and_grad(critic_loss_fn)
            critic_loss, critic_grads = critic_loss_grad_fn(state.critic_params,
                                                            state.target_critic_params,
                                                            state.target_actor_params,
                                                            critic_key,
                                                            batch)
            
            critic_update, new_critic_opt_state = critic_opt.update(critic_grads, state.critic_opt_state)
            new_critic_params = optax.apply_updates(state.critic_params, critic_update)
            
            metrics = {'critic_loss': critic_loss}
            
            # update actor + targets (if needed)
            def update_actor_and_targets(_):
                actor_loss_grad_fn = jax.value_and_grad(actor_loss_fn)
                actor_loss, actor_grads = actor_loss_grad_fn(state.actor_params,
                                                             new_critic_params,
                                                             batch)
                
                actor_update, new_actor_opt_state = actor_opt.update(actor_grads, state.actor_opt_state)
                new_actor_params = optax.apply_updates(state.actor_params, actor_update)
                
                new_target_actor_params = update_target(new_actor_params, state.target_actor_params, tau)
                new_target_critic_params = update_target(new_critic_params, state.target_critic_params, tau)
                
                return actor_loss, new_actor_params, new_actor_opt_state, new_target_actor_params, new_target_critic_params
            
            def do_nothing(_):
                return jnp.inf, state.actor_params, state.actor_opt_state, state.target_actor_params, state.target_critic_params
            
            actor_loss, new_actor_params, new_actor_opt_state, new_target_actor_params, new_target_critic_params = jax.lax.cond(
                step % target_update_freq == 0,
                update_actor_and_targets,
                do_nothing,
                operand=None
            )
            
            new_state = TD3BCState(
                actor_params=new_actor_params,
                target_actor_params=new_target_actor_params,
                critic_params=new_critic_params,
                target_critic_params=new_target_critic_params,
                actor_opt_state=new_actor_opt_state,
                critic_opt_state=new_critic_opt_state,
                rng_key=state_key
            )
            
            metrics.update({'actor_loss': actor_loss})
            return new_state, metrics
        
        self._act = jax.jit(act)
        self._update = jax.jit(update)