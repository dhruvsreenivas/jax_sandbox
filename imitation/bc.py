import jax
import haiku as hk
from common.neural_net import *
from common.dataset import ExpertBatch
from common.utils import get_opt_class
import numpy as np
import optax

class BC:
    def __init__(self, cfg):
        if cfg.continuous:
            self.policy_fn = ContinuousPolicy(cfg)
        else:
            self.policy_fn = DiscretePolicy(cfg)
         
        # transform to allow init + apply
        self.policy = hk.transform(lambda x: self.policy_fn(x))
        
        # optimizer
        self.optimizer = get_opt_class(cfg.opt)(learning_rate=cfg.lr)
        
        # rng sequence
        self.rng_seq = hk.PRNGSequence(cfg.seed)
        
        # initialization of params and opt state
        self.params = self.policy.init(next(self.rng_seq), np.zeros(1, *cfg.obs_shape))
        self.opt_state = self.optimizer.init(self.params)

    def learn(self, batch: ExpertBatch):
        # define loss fn and apply updates w.r.t parameters
        def loss_fn(params, rng_key, batch):
            dist = self.policy.apply(params, rng_key, batch.states)
            lp = dist.log_prob(batch.actions)
            loss = -lp.mean()
            return loss
        
        loss = loss_fn(self.params, next(self.rng_seq), batch)
        grads = jax.grad(loss)(self.params, batch)
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state)
        new_params = optax.apply_updates(self.params, updates)
        
        self.params = new_params
        self.opt_state = new_opt_state
        
        return loss.item()