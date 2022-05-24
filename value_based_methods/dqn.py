from common.neural_net import DiscreteQNetwork, ContinuousQNetwork
from common.utils import *
from common.dataset import TransitionBatch
import haiku as hk
import numpy as np
import jax
import rlax

batch_q_learning_fn = jax.vmap(rlax.q_learning)

class DQN:
    def __init__(self, cfg):
        self.gamma = cfg.gamma
        
        assert not cfg.continuous, 'DQN only works with discrete action spaces.'
        # initialize q net and target q net
        self.qnet_fn = DiscreteQNetwork(cfg)
        self.target_qnet_fn = DiscreteQNetwork(cfg)
        
        # transform for init/apply
        self.qnet = hk.transform(lambda x: self.qnet_fn(x))
        self.target_qnet = hk.transform(lambda x: self.target_qnet(x))
        
        # optimizer
        self.opt = get_opt_class(cfg.opt)(learning_rate=cfg.lr)
        self.opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), self.opt)
        
        # rng sequence
        self.rng_seq = hk.PRNGSequence(cfg.seed)
        
        # online + target params
        rng_key = next(self.rng_seq)
        self.online_params = self.qnet.init(rng_key, np.zeros(1, *cfg.obs_shape))
        self.target_params = self.target_qnet.init(rng_key, np.zeros(1, *cfg.obs_shape))
        
        # opt state initialization
        self.opt_state = self.optimizer.init(self.online_params)
        
    def target_update(self, tau=0.9):
        # TODO: make sure you know how to change parameters
        self.target_params = tau * self.target_params + (1 - tau) * self.online_params
        
    def learn(self, batch: TransitionBatch):
        # define loss fn, then take gradient and step
        def loss_fn(params, target_params, rng_key, batch):
            online_rng_key, target_rng_key = jax.random.split(rng_key, 2)
            targets = self.target_qnet.apply(target_params, target_rng_key, batch.next_states)
            outputs = self.qnet.apply(params, online_rng_key, batch.states)
            
            td_errors = batch_q_learning_fn(
                outputs,
                batch.actions,
                batch.rewards,
                self.gamma * jnp.ones_like(batch.rewards),
                targets
            )
            
            losses = rlax.l2_loss(td_errors)
            
            return jnp.mean(losses)

        rng_key = next(self.rng_seq)
        loss = loss_fn(self.online_params, self.target_params, rng_key, batch)
        grads = jax.grad(loss_fn)(self.online_params, self.target_params, rng_key, batch)
        updates, new_opt_state = self.opt.update(grads, self.opt_state)
        new_params = optax.apply_updates(self.online_params, updates)
        
        self.online_params = new_params
        self.opt_state = new_opt_state
        
        return loss.item()