from common.neural_net import *
from common.dataset import ExpertBatch, TransitionBatch
from common.utils import *
from policy_gradient.ddpg import DDPG
from policy_gradient.sac import SAC

class GAIL:
    '''Generative adversarial imitation learning.'''
    def __init__(self, cfg):
        self.is_continuous = cfg.continuous
        
        # rng seq
        self.rng_seq = hk.PRNGSequence(cfg.seed)
        
        # discriminator initialization
        if cfg.continuous:
            self.disc = hk.without_apply_rng(hk.transform(lambda x, a: ContinuousQNetwork(cfg)(x, a)))
            self.disc_params = self.disc.init(next(self.rng_seq), jnp.zeros((1, np.prod(cfg.obs_shape))), jnp.ones((1, cfg.n_actions)))
        else:
            self.disc = hk.without_apply_rng(hk.transform(lambda x: DiscreteQNetwork(x)))
            self.disc_params = self.disc.init(next(), jnp.zeros((1, *cfg.obs_shape)))
        
        # optimizer
        self.disc_opt = get_opt_class(cfg.opt)(learning_rate=cfg.lr)
        if cfg.clip_grad_norm:
            self.disc_opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), self.disc_opt)
        
        # initialize discriminator opt state
        self.disc_opt_state = self.disc_opt.init(self.disc_params)
        
        # TODO create policy (currently only supports SAC/DDPG)
        if cfg.rl_algo == 'ddpg':
            self.policy = DDPG(cfg)
        else:
            self.policy = SAC(cfg)
    
    def learn(self, expert_batch: ExpertBatch, policy_batch: TransitionBatch):
        # TODO: once policy is created, run
        pass