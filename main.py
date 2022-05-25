from common import *
from imitation import *
from policy_gradient import *
from value_based_methods import *
import hydra

class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.alg == 'bc':
            self.learner = bc.BC(cfg.bc)
        elif cfg.alg == 'gail':
            self.learner = gail.GAIL(cfg.gail)
        elif cfg.alg == 'ddpg':
            self.learner = ddpg.DDPG(cfg.ddpg)
        elif cfg.alg == 'sac':
            self.learner = sac.SAC(cfg.sac)
        elif cfg.alg == 'dqn':
            self.learner = dqn.DQN(cfg.dqn)
        else:
            raise ValueError('RL algorithm not implemented yet.')

@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    pass