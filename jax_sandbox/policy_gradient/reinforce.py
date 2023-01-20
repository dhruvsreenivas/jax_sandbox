import jax
import haiku as hk
from typing import NamedTuple, Tuple, Dict

from jax_sandbox.common.nets import *
from jax_sandbox.common.dataset import TransitionBatch
from jax_sandbox.common.utils import opt_class
from jax_sandbox.policy_gradient.pg_utils import *

class REINFORCEState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    rng_key: jax.random.PRNGKey

class REINFORCE:
    def __init__(self, cfg):
        # set up nets
        if cfg.img_input:
            channels = list(cfg.channels)
            kernels = list(cfg.kernels)
            strides = list(cfg.strides)
            
        assert not cfg.deterministic, "REINFORCE doesn't use a deterministic policy."
        
        # for continuous vs. discrete tasks (I think REINFORCE can do both)
        if cfg.img_input:
            if cfg.baseline:
                def actor_fn(s):
                    net = hk.Sequential([
                        ConvTorso(channels, kernels, strides, act=cfg.activation, activate_final=cfg.activate_final),
                        PolicyValueHead(cfg.action_shape, not cfg.continuous, cfg.softplus, cfg.min_std)
                    ])
                    return net(s)
            else:
                if cfg.continuous:
                    def actor_fn(s):
                        net = hk.Sequential([
                            ConvTorso(channels, kernels, strides, act=cfg.activation, activate_final=cfg.activate_final),
                            LinearGaussian(cfg.action_shape, not cfg.continuous, cfg.softplus, cfg.min_std)
                        ])
                        return net(s)
                else:
                    def actor_fn(s):
                        net = hk.Sequential([
                            ConvTorso(channels, kernels, strides, act=cfg.activation, activate_final=cfg.activate_final),
                            LinearCategorical(cfg.action_shape, not cfg.continuous, cfg.softplus, cfg.min_std)
                        ])
                        return net(s)
        else:
            if cfg.baseline:
                def actor_fn(s):
                    net = hk.Sequential([
                        MLP(cfg.hidden_sizes, act=cfg.activation, activate_final=cfg.activate_final, output_act=cfg.output_act, use_ln=cfg.use_ln),
                        PolicyValueHead(cfg.action_shape, not cfg.continuous, cfg.softplus, cfg.min_std)
                    ])
                    return net(s)
            else:
                if cfg.continuous:
                    def actor_fn(s):
                        net = hk.Sequential([
                            MLP(cfg.hidden_sizes, act=cfg.activation, activate_final=cfg.activate_final, output_act=cfg.output_act, use_ln=cfg.use_ln),
                            LinearGaussian(cfg.action_shape, cfg.softplus, cfg.min_std)
                        ])
                        return net(s)
                else:
                    def actor_fn(s):
                        net = hk.Sequential([
                            MLP(cfg.hidden_sizes, act=cfg.activation, activate_final=cfg.activate_final, output_act=cfg.output_act, use_ln=cfg.use_ln),
                            LinearCategorical(cfg.action_shape)
                        ])
                        return net(s)
        
        actor = hk.without_apply_rng(hk.transform(actor_fn))
        
        # init
        key = jax.random.PRNGKey(cfg.seed)
        init_key, state_key = jax.random.split(key)
        params = actor.init(init_key, batched_zeros_like(cfg.obs_shape))
        
        # optimizer
        opt = opt_class(cfg.optim)(learning_rate=cfg.lr)
        opt_state = opt.init(params)
        
        self._state = REINFORCEState(
            params=params,
            opt_state=opt_state,
            rng_key=state_key
        )
        
        # hparams
        gamma = cfg.gamma
        
        
        def act(state: jnp.ndarray, eval_mode: bool) -> jnp.ndarray:
            if cfg.baseline:
                policy_dist, _ = actor.apply(self._state.params, state)
            else:
                policy_dist = actor.apply(self._state.params, state)
            
            key, subkey = jax.random.split(self._state.rng_key)
            mean_action = policy_dist.mode()
            sampled_action = policy_dist.sample(seed=subkey)
            action = jnp.where(eval_mode, mean_action, sampled_action)
            
            self._state._replace(rng_key=key)
            return action
        
        @jax.jit
        def loss_fn(params: hk.Params, batch: TransitionBatch):
            '''Loss function for online RL--make sure batch is online batch!'''
            if cfg.baseline:
                policy_dist, value = actor.apply(params, batch.states)
                value = jnp.squeeze(value)
            else:
                policy_dist = actor.apply(params, batch.states)
            
            log_probs = policy_dist.log_prob(batch.actions).sum(-1) # (B,)
            rtgs = returns_to_go(batch, gamma)
            if cfg.baseline:
                rtgs = rtgs - value
            
            loss = (rtgs * log_probs).mean()
            return loss
        
        def update(state: REINFORCEState, batch: TransitionBatch, step: int) -> Tuple[REINFORCEState, Dict]:
            del step
            
            loss_grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = loss_grad_fn(state.params, batch)
            
            update, new_opt_state = opt.update(grads, state.opt_state)
            new_params = optax.apply_updates(state.params, update)
            
            new_state = REINFORCEState(
                params=new_params,
                opt_state=new_opt_state,
                rng_key=state.rng_key
            )
            metrics = {'pg_loss': loss}
            return new_state, metrics
        
        self._act = jax.jit(act)
        self._update = jax.jit(update)