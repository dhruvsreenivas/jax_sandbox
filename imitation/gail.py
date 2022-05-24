from common.neural_net import *
from common.dataset import ExpertBatch, TransitionBatch
from common.utils import *

class GAIL:
    '''Generative adversarial imitation learning.'''
    def __init__(self, cfg):
        
        # discriminator initialization
        if cfg.continuous:
            self.disc_fn = ContinuousQNetwork(cfg)
        else:
            self.disc_fn = DiscreteQNetwork(cfg)
            
        # transform for init + apply
        self.disc = hk.transform(lambda x: self.disc_fn(x)) # hopefully can just pass in the function sooner or later
        
        # optimizer
        self.disc_opt = get_opt_class(cfg.opt)(learning_rate=cfg.lr)
        self.disc_opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), self.disc_opt)
        
        # rng seq
        self.rng_seq = hk.PRNGSequence(cfg.seed)
        
        # initialize discriminator params + opt state
        self.disc_params = self.disc.init(next(self.rng_seq), jnp.zeros(1, cfg.obs_shape + cfg.n_actions))
        self.disc_opt_state = self.disc_opt.init(self.disc_params)
        
        # TODO create policy
    
    def learn(self, expert_batch: ExpertBatch, policy_batch: TransitionBatch):
        # TODO: once policy is created, run
        pass