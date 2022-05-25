import jax
import jax.numpy as jnp
import haiku as hk
import optax
import rlax
from common.neural_net import *
from common.dataset import TransitionBatch
from common.utils import get_opt_class

class SAC:
    '''Continuous soft-actor critic.'''
    def __init__(self, cfg):
        policy_fn = Policy(cfg)
        self.policy = hk.transform(lambda x: policy_fn(x))
        
        # q1, q2 function params (TODO: add code for shared policy + value fn in all appropriate algorithms)
        qnet1_fn = ContinuousQNetwork(cfg)
        qnet2_fn = ContinuousQNetwork(cfg)
        # deterministic functions, so rng is not needed when transforming
        self.qnet1 = hk.without_apply_rng(hk.transform(lambda x, a: qnet1_fn(x, a)))
        self.qnet2 = hk.without_apply_rng(hk.transform(lambda x, a: qnet2_fn(x, a)))
        
        # optimizers
        self.policy_opt = get_opt_class(cfg.opt)(learning_rate=cfg.lr)
        self.q1_opt = get_opt_class(cfg.opt)(learning_rate=cfg.lr)
        self.q2_opt = get_opt_class(cfg.opt)(learning_rate=cfg.lr)
        if cfg.clip_grad_norm:
            self.policy_opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), self.policy_opt)
            self.q1_opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), self.q1_opt)
            self.q2_opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), self.q2_opt)
        
        # rng sequence
        self.rng_seq = hk.PRNGSequence(cfg.seed)
        
        # initialization
        rng = next(self.rng_seq)
        self.policy_params = self.policy.init(rng, jnp.zeros(1, *cfg.obs_shape))
        self.policy_opt_state = self.policy_opt.init(self.policy_params)
        
        rng = next(self.rng_seq)
        self.q1_params = self.target_q1_params = self.qnet1.init(rng, jnp.zeros(1, *cfg.obs_shape))
        self.q1_opt_state = self.q1_opt.init(self.q1_params)
        
        rng = next(self.rng_seq)
        self.q2_params = self.target_q2_params = self.qnet2.init(rng, jnp.zeros(1, *cfg.obs_shape))
        self.q2_opt_state = self.q2_opt.init(self.q2_params)
        
        # additional things
        self.gamma = cfg.gamma
        self.temp = cfg.temp
        
    def sample_actions(self, params, states, return_logprob=False):
        action_dist = self.policy.apply(params, next(self.rng_seq), states)
        a, lps = action_dist.sample_and_log_prob(seed=next(self.rng_seq))
        if return_logprob:
            return a, lps
        return a
    
    def learn(self, batch: TransitionBatch):
        # def value and policy fn losses and do damage
        def value_loss_fn(q1_params, q2_params, target_q1_params, target_q2_params, rng_key, batch):
            rng_key, rng_key2 = jax.random.split(rng_key) # two keys for two different q functions i think, although for apply probably not
            a_tp1, lps = self.sample_actions(batch.next_states, return_logprob=True)
            sas_tp1 = jnp.concatenate((batch.next_states, a_tp1), axis=1)
            q1_tp1 = self.qnet1.apply(target_q1_params, rng_key, sas_tp1)
            q2_tp1 = self.qnet2.apply(target_q2_params, rng_key2, sas_tp1)
            targets = batch.rewards + self.gamma * (1.0 - batch.dones) * (jnp.minimum(q1_tp1, q2_tp1) - self.temp * lps)
            
            rng_key, rng_key2 = jax.random.split(rng_key)
            sas = jnp.concatenate((batch.states, batch.actions), axis=1)
            q1s = self.qnet1.apply(q1_params, rng_key, sas)
            q2s = self.qnet2.apply(q2_params, rng_key2, sas)
            
            loss = rlax.l2_loss(q1s, targets) + rlax.l2_loss(q2s, targets)
            return loss.mean()
        
        def policy_loss_fn(policy_params, q1_params, q2_params, rng_key, batch):
            actions, lps = self.sample_actions(policy_params, batch.states, return_logprob=True)
            sas = jnp.concatenate((batch.states, actions), axis=1)
            rng_key, rng_key2 = jax.random.split(rng_key) # same idea here as above
            q1s = self.qnet1.apply(q1_params, rng_key, sas)
            q2s = self.qnet2.apply(q2_params, rng_key2, sas)
            qs = jnp.minimum(q1s, q2s)
            ent_reg_qs = qs - self.temp * lps
            
            return -ent_reg_qs.mean()
        
        # q function update
        rng_key = next(self.rng_seq)
        v_loss = value_loss_fn(self.q1_params, self.q2_params, self.target_q1_params, self.target_q2_params, rng_key, batch)
        v_grads = jax.grad(value_loss_fn)(self.q1_params, self.q2_params, self.target_q1_params, self.target_q2_params, rng_key, batch)
        
        updates1, new_opt_state1 = self.q1_opt.update(v_grads, self.q1_opt_state)
        new_q1_params = optax.apply_updates(self.q1_params, updates1)
        self.q1_params = new_q1_params
        self.q1_opt_state = new_opt_state1
        
        updates2, new_opt_state2 = self.q2_opt.update(v_grads, self.q2_opt_state)
        new_q2_params = optax.apply_updates(self.q2_params, updates2)
        self.q2_params = new_q2_params
        self.q2_opt_state = new_opt_state2
        
        # policy update
        rng_key = next(self.rng_seq)
        p_loss = policy_loss_fn(self.policy_params, self.q1_params, self.q2_params, rng_key, batch)
        p_grads = jax.grad(policy_loss_fn)(self.policy_params, self.q1_params, self.q2_params, rng_key, batch)
        updates, new_opt_state = self.policy_opt.update(p_grads, self.policy_opt_state)
        new_params = optax.apply_updates(self.policy_params, updates)
        self.policy_params = new_params
        self.policy_opt_state = new_opt_state
        
        return {
            'policy loss': p_loss.item(),
            'value loss': v_loss.item()
        }
        
        
        
            
    
    