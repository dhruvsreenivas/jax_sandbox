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
        self.policy = hk.transform(lambda x: Policy(cfg)(x))
        self.qnet = hk.without_apply_rng(hk.transform(lambda x, a: ContinuousQNetwork(cfg)(x, a)))
        
        # optimizers (gradient norm needed?)
        self.policy_opt = get_opt_class(cfg.opt)(learning_rate=cfg.lr)
        self.q_opt = get_opt_class(cfg.opt)(learning_rate=cfg.lr)
        if cfg.clip_grad_norm:
            self.policy_opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), self.policy_opt)
            self.q_opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), self.q_opt)
        
        # rng sequence
        self.rng_seq = hk.PRNGSequence(cfg.seed)
        
        # initialization of params and opt state (policy + q net)
        rng = next(self.rng_seq)
        self.policy_params = self.target_params = self.policy.init(rng, jnp.zeros((1, *cfg.obs_shape)))
        self.policy_opt_state = self.policy_opt.init(self.policy_params)

        rng = next(self.rng_seq)
        self.q_params = self.target_q_params = self.qnet.init(rng, jnp.zeros((1, np.prod(cfg.obs_shape))), jnp.ones((1, cfg.n_actions)))
        self.q_opt_state = self.q_opt.init(self.q_params)
        
        self.gamma = cfg.gamma
        
    def sample_actions(self, states):
        a_dist = self.policy.apply(self.target_params, next(self.rng_seq), states)
        return a_dist.sample(seed=hk.next_rng_key())
    
    def get_mean_action(self, params, states):
        a_dist = self.policy.apply(params, next(self.rng_seq), states)
        return a_dist.loc
        
    def learn(self, batch: TransitionBatch):
        # compute loss fn
        def value_loss_fn(params, target_params, batch):
            # purely deterministic, no rng key needed here
            a_tp1 = self.sample_actions(batch.next_states)
            qs_tp1 = self.qnet.apply(target_params, batch.next_states, a_tp1)
            targets = batch.rewards + self.gamma * (1.0 - batch.dones) * qs_tp1
            
            qs = self.qnet.apply(params, batch.states, batch.actions)
            
            loss = rlax.l2_loss(qs, targets)
            return loss.mean()
        
        def policy_loss_fn(policy_params, q_params, rng_key, batch):
            mean_actions = self.get_mean_action(policy_params, batch.states)
            qs = self.qnet.apply(q_params, rng_key, batch.states, mean_actions)
            return -qs.mean()
        
        # update value net first
        rng_key = next(self.rng_seq)
        v_loss = value_loss_fn(self.q_params, self.target_q_params, batch)
        v_grads = jax.grad(value_loss_fn)(self.q_params, self.target_q_params, batch)
        updates, new_opt_state = self.q_opt.update(v_grads, self.q_opt_state)
        new_params = optax.apply_updates(self.q_params, updates)
        
        self.q_params = new_params
        self.q_opt_state = new_opt_state
        
        # now update policy
        rng_key = next(self.rng_seq)
        p_loss = policy_loss_fn(self.q_params, rng_key, batch)
        p_grads = jax.grad(policy_loss_fn)(self.policy_params, self.q_params, rng_key, batch)
        updates, new_opt_state = self.policy_opt.update(p_grads, self.policy_opt_state)
        new_params = optax.apply_updates(self.policy_params, updates)
        
        return {
            'policy loss': p_loss.item(),
            'value loss': v_loss.item()
        }