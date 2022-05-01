from common.neural_net import DiscreteQNetwork, ContinuousQNetwork
from common.utils import *
from common.dataset import TransitionBatch
import haiku as hk
import numpy as np
import jax

class DQN:
    def __init__(self, cfg):
        
        # initialize q net and target q net
        if cfg.continuous:
            self.qnet_fn = ContinuousQNetwork(cfg)
            self.target_qnet_fn = ContinuousQNetwork(cfg)
        else:
            self.qnet_fn = DiscreteQNetwork(cfg)
            self.target_qnet_fn = DiscreteQNetwork(cfg)
        
        # transform for init/apply
        self.qnet = hk.transform(lambda x: self.qnet_fn(x))
        self.target_qnet = hk.transform(lambda x: self.target_qnet(x))
        
        # optimizer
        self.opt = get_opt_class(cfg.opt)(learning_rate=cfg.lr)
        
        # rng sequence
        self.rng_seq = hk.PRNGSequence(cfg.seed)
        
        # online + target params
        rng_key = next(self.rng_seq)
        self.online_params = self.qnet.init(rng_key, np.zeros(1, *cfg.obs_shape))
        self.target_params = self.target_qnet.init(rng_key, np.zeros(1, *cfg.obs_shape))