import jax
import haiku as hk
import optax
from typing import Tuple, NamedTuple, Dict

from jax_sandbox.common.nets import *
from jax_sandbox.common.dataset import ExpertBatch
from jax_sandbox.common.utils import *

class BCState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    rng_key: jax.random.PRNGKey

class BC:
    def __init__(self, cfg):
        # define policy fn
        if cfg.img_input:
            channels = list(cfg.channels)
            kernels = list(cfg.kernels)
            strides = list(cfg.strides)
            
            if not cfg.continuous:
                if cfg.deterministic:
                    def policy_fn(s):
                        net = hk.Sequential([
                            ConvTorso(channels, kernels, strides, act=cfg.activation, activate_final=cfg.activate_final),
                            hk.Linear(cfg.action_shape)
                        ])
                        return net(s)
                else:
                    def policy_fn(s):
                        net = hk.Sequential([
                            ConvTorso(channels, kernels, strides, act=cfg.activation, activate_final=cfg.activate_final),
                            LinearGaussian(cfg.action_shape, softplus=cfg.softplus, min_std=cfg.min_std)
                        ])
                        return net(s)
            else:
                def policy_fn(s):
                    net = hk.Sequential([
                        ConvTorso(channels, kernels, strides, act=cfg.activation, activate_final=cfg.activate_final),
                        LinearCategorical(cfg.action_shape)
                    ])
                    return net(s)
        else:
            if cfg.continuous:
                if cfg.deterministic:
                    def policy_fn(s):
                        net = hk.Sequential([
                            MLP(cfg.hidden_sizes, act=cfg.activation, activate_final=cfg.activate_final, use_ln=cfg.use_ln),
                            hk.Linear(cfg.action_shape)
                        ])
                        return net(s)
                else:
                    def policy_fn(s):
                        net = hk.Sequential([
                            MLP(cfg.hidden_sizes, act=cfg.activation, activate_final=cfg.activate_final, use_ln=cfg.use_ln),
                            LinearGaussian(cfg.action_shape)
                        ])
                        return net(s)
            else:
                def policy_fn(s):
                    net = hk.Sequential([
                        MLP(cfg.hidden_sizes, act=cfg.activation, activate_final=cfg.activate_final, use_ln=cfg.use_ln),
                        LinearCategorical(cfg.action_shape)
                    ])
                    return net(s)
            
        actor = hk.without_apply_rng(hk.transform(policy_fn))
        
        # set up state
        key = jax.random.PRNGKey(cfg.seed)
        actor_key, state_key = jax.random.split(key)
        
        params = actor.init(actor_key, batched_zeros_like(cfg.obs_shape))
        
        opt = opt_class(cfg.optim)(learning_rate=cfg.lr)
        opt_state = opt.init(params)
        
        self._state = BCState(
            params=params,
            opt_state=opt_state,
            rng_key=state_key
        )
        
        # define act, loss, update functions
        
        def act(state: jnp.ndarray, eval_mode: bool) -> jnp.ndarray:
            if cfg.deterministic:
                action = actor.apply(self._state.params, state)
            else:
                action_dist = actor.apply(self._state.params, state)
                if eval_mode:
                    action = action_dist.mode()
                else:
                    key, subkey = jax.random.split(self._state.rng_key)
                    action = action_dist.sample(seed=subkey)
                    self._state = self._state._replace(rng_key=key)
                    
            return action
        
        @jax.jit
        def loss_fn(params: hk.Params, batch: ExpertBatch) -> jnp.ndarray:
            out = actor.apply(params, batch.states)
            if cfg.deterministic:
                loss = jnp.mean(jnp.square(out - batch.actions))
            else:
                lps = out.log_prob(batch.actions)
                loss = -lps.mean()
            
            return loss
        
        def update(state: BCState, batch: ExpertBatch, step: int) -> Tuple[BCState, Dict]:
            del step
            
            loss_grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = loss_grad_fn(state.params, batch)
            
            update, new_opt_state = opt.update(grads, state.opt_state)
            new_params = optax.apply_updates(state.params, update)
            
            new_state = BCState(
                params=new_params,
                opt_state=new_opt_state,
                rng_key=state.rng_key
            )
            metrics = {'bc_loss': loss}
            
            return new_state, metrics
        
        self._act = jax.jit(act)
        self._update = jax.jit(update)