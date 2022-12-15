from common.nets import *
from common.utils import *
from common.dataset import TransitionBatch
import haiku as hk
import jax
import rlax
import optax
from typing import NamedTuple, Tuple, Dict

class DQNState(NamedTuple):
    online_params: hk.Params
    target_params: hk.Params
    opt_state: optax.OptState
    rng_key: jax.random.PRNGKey

class DQN:
    def __init__(self, cfg):
        if cfg.img_input:
            channels = list(cfg.channels)
            kernels = list(cfg.kernels)
            strides = list(cfg.strides)
            
        assert not cfg.continuous, "DQN doesn't work for continuous spaces."
        
        if cfg.img_input:
            def q_fn(s):
                net = hk.Sequential([
                    ConvTorso(channels, kernels, strides, act=cfg.activation, activate_final=cfg.activate_final),
                    hk.Linear(cfg.action_shape)
                ])
                return net(s)
        else:
            def q_fn(s):
                net = hk.Sequential([
                    MLP(cfg.hidden_sizes, act=cfg.activation, activate_final=cfg.activate_final, use_ln=cfg.use_ln),
                    hk.Linear(cfg.action_shape)
                ])
                return net(s)
        
        q = hk.without_apply_rng(hk.transform(q_fn))
        
        # init parameters
        key = jax.random.PRNGKey(cfg.seed)
        param_key, state_key = jax.random.split(key)
        online_params = target_params = q.init(param_key, batched_zeros_like(cfg.obs_shape))
        
        opt = opt_class(cfg.optim)(learning_rate=cfg.lr)
        opt_state = opt.init(online_params)
        
        self._state = DQNState(
            online_params=online_params,
            target_params=target_params,
            opt_state=opt_state,
            rng_key=state_key
        )
        
        gamma = cfg.gamma
        eps = cfg.eps
        double_q = cfg.double_q
        target_update_freq = cfg.target_update_freq
        tau = cfg.tau
        
        # define functions of interest
        @jax.jit
        def act(state: jnp.ndarray, eval_mode: bool) -> jnp.ndarray:
            q_vals = q.apply(self._state.online_params, state)
            
            if eval_mode:
                action = jnp.argmax(q_vals, axis=-1)
            else:
                # do epsilon greedy sampling
                key, subkey, subsubkey = jax.random.split(self._state.rng_key, 3)
                val = jax.random.uniform(subkey)
                if val < eps:
                    # do exploration
                    action = jax.random.randint(subsubkey, shape=q_vals.shape[0], minval=0, maxval=cfg.action_shape)
                else:
                    action = jnp.argmax(q_vals, axis=-1)
            
                self._state = self._state._replace(rng_key=key)
            
            return action

        @jax.jit
        def loss_fn(online_params: hk.Params, target_params: hk.Params, batch: TransitionBatch) -> jnp.ndarray:
            # get targets
            next_target_q = q.apply(target_params, batch.next_states) # [B, A]
            
            if double_q:
                next_online_q = q.apply(online_params, batch.next_states) # [B, A]
                next_actions = jnp.argmax(next_online_q, axis=-1) # [B]
                next_actions = jax.nn.one_hot(next_actions, num_classes=cfg.action_shape) # [B, A]
                next_v = jnp.sum(next_target_q * next_actions, axis=-1)
            else:
                next_v = jnp.argmax(next_target_q, axis=-1)
                
            targets = batch.rewards + gamma * (1.0 - batch.dones) * next_v
            targets = jax.lax.stop_gradient(targets)
            
            # online q vals
            curr_q = q.apply(online_params, batch.states)
            curr_q = curr_q[jnp.arange(curr_q.shape[0]), batch.actions]
            
            loss = rlax.l2_loss(curr_q, targets)
            return loss
        
        def update(state: DQNState, batch: TransitionBatch, step: int) -> Tuple[DQNState, Dict]:
            loss_grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = loss_grad_fn(state.online_params, state.target_params, batch)

            update, new_opt_state = opt.update(grads, state.opt_state)
            new_params = optax.apply_updates(state.online_params, update)
            
            if step % target_update_freq == 0:
                target_params = update_target(new_params, target_params, tau)
            else:
                target_params = state.target_params
            
            new_state = DQNState(
                online_params=new_params,
                target_params=target_params,
                opt_state=new_opt_state,
                rng_key=state.rng_key
            )
            metrics = {'q_loss': loss}
            
            return new_state, metrics
        
        self._act = jax.jit(act)
        self._update = jax.jit(update)