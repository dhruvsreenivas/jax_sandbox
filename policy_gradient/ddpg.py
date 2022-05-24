from typing import ParamSpecArgs
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import rlax
from common.neural_net import *
from common.dataset import TransitionBatch
from common.utils import get_opt_class

class DDPG:
    def __init__(self, cfg):
        '''DDPG algorithm for continuous control.'''
        policy_fn = Policy(cfg)
        self.policy = hk.transform(lambda x: policy_fn(x))
        self.target_policy = hk.transform(lambda x: policy_fn(x))
        
        qnet_fn = ContinuousQNetwork(cfg)
        self.qnet = hk.transform(lambda x: qnet_fn(x))
        self.target_qnet = hk.transform(lambda x: qnet_fn(x))
        
        # optimizers (gradient norm needed?)
        self.policy_opt = get_opt_class(cfg.opt)(learning_rate=cfg.lr)
        self.policy_opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), self.policy_opt)
        self.q_opt = get_opt_class(cfg.opt)(learning_rate=cfg.lr)
        self.q_opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), self.q_opt)
        
        # rng sequence
        self.rng_seq = hk.PRNGSequence(cfg.seed)
        
        # initialization of params and opt state (policy + q net)
        rng = next(self.rng_seq)
        self.policy_params = self.policy.init(rng, jnp.zeros(1, *cfg.obs_shape))
        self.target_params = self.target_policy.init(rng, jnp.zeros(1, *cfg.obs_shape))
        self.policy_opt_state = self.policy_opt.init(self.policy_params)

        rng = next(self.rng_seq)
        self.q_params = self.qnet.init(rng, jnp.zeros(1, *cfg.obs_shape))
        self.target_q_params = self.target_qnet.init(rng, jnp.zeros(1, *cfg.obs_shape))
        self.q_opt_state = self.q_opt.init(self.q_params)
        
        self.gamma = cfg.gamma
        
    def sample_target_actions(self, states):
        a_dist = self.target_policy.apply(self.target_params, next(self.rng_seq), states)
        return a_dist.sample(seed=hk.next_rng_key())
        
    def learn(self, batch: TransitionBatch):
        # compute loss fn
        def value_loss_fn(params, rng_key, batch):
            a_tp1 = self.sample_target_actions(batch.next_states)
            qs_tp1 = self.qnet.apply(params, rng_key, [batch.next_states, a_tp1])
            targets = batch.rewards + self.gamma * (1.0 - batch.dones) * jnp.max(qs_tp1, axis=1)
            
            qs = self.qnet.apply(params, rng_key, batch.states)
            loss = rlax.l2_loss(qs, targets)
        
            return loss.mean()
        
        def policy_loss_fn(params, rng_key, batch):
            qs = self.qnet.apply(params, rng_key, [batch.states, batch.actions])
            return -qs.mean()
        
        # update value net first
        v_loss = value_loss_fn(self.q_params, next(self.rng_seq), batch)
        v_grads = jax.grad(value_loss_fn)(self.q_params, batch)
        updates, new_opt_state = self.q_opt.update(v_grads, self.q_opt_state)
        new_params = optax.apply_updates(self.q_params, updates)
        
        self.q_params = new_params
        self.q_opt_state = new_opt_state
        
        # now update policy
        p_loss = policy_loss_fn(self.policy_params, next(self.rng_seq), batch)
        p_grads = jax.grad(policy_loss_fn)(self.policy_params)
        updates, new_opt_state = self.policy_opt.update(p_grads, self.policy_opt_state)
        new_params = optax.apply_updates(self.policy_params, updates)
        
        return {
            'policy loss': p_loss.item(),
            'value loss': v_loss.item()
        }