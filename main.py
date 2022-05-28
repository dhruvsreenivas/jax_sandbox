from common import *
from imitation import *
from policy_gradient import *
from value_based_methods import *
import hydra

class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.alg == 'bc':
            self.learner = bc.BC(cfg)
        elif cfg.alg == 'gail':
            self.learner = gail.GAIL(cfg)
        elif cfg.alg == 'ddpg':
            self.learner = ddpg.DDPG(cfg)
        elif cfg.alg == 'sac':
            self.learner = sac.SAC(cfg)
        elif cfg.alg == 'dqn':
            self.learner = dqn.DQN(cfg)
        else:
            raise ValueError('RL algorithm not implemented yet.')

@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    pass