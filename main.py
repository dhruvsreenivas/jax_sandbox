import warnings
warnings.filterwarnings("ignore")

from common import *
from imitation import *
from policy_gradient import *
from value_based_methods import *
import hydra

import envs.dmc as dmc
from utils import *

def make_env(env_name, seed):
    if env_name in ['walker_walk', 'cheetah_run', 'humanoid_walk', 'finger_turn_hard', 'cartpole_swingup', 'hopper_hop', 'quadruped_walk', 'reacher_hard']:
        env = dmc.make(env_name, seed=seed)
    else:
        env = make_gym_env(env_name, seed)
        
    return env

def get_observation_action_spec(env):
    if hasattr(env, 'observation_space'):
        obs_shape = env.observation_space.shape
        
        if isinstance(env.action_space, gym.spaces.Box):
            action_shape = env.action_space.shape[0]
        else:
            action_shape = env.action_space.n
    else:
        obs_shape = env.observation_spec().shape
        action_shape = env.action_spec().shape
    
    return obs_shape, action_shape

OFFLINE_ALGOS = ['cql', 'td3_bc', 'milo']

class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.setup()
        print('done setting up')
        
        if cfg.alg == 'bc':
            self.learner = bc.BC(cfg)
        elif cfg.alg == 'gail':
            self.learner = gail.GAIL(cfg)
        elif cfg.alg == 'reinforce':
            self.learner = reinforce.REINFORCE(cfg)
        elif cfg.alg == 'ddpg':
            self.learner = ddpg.DDPG(cfg)
        elif cfg.alg == 'sac':
            self.learner = sac.SAC(cfg)
        elif cfg.alg == 'dqn':
            self.learner = dqn.DQN(cfg)
        else:
            raise ValueError('RL algorithm not implemented yet.')
        
    def setup(self):
        # setup env stuff and fill in unknown cfg values
        self.train_env = make_env(self.cfg.task, self.cfg.seed)
        self.eval_env = make_env(self.cfg.task, self.cfg.seed)
        
        self.cfg.obs_shape, self.cfg.action_shape = get_observation_action_spec(self.train_env)
        self.cfg.continuous = is_discrete(self.cfg.task)
        self.cfg.img_input = len(self.cfg.obs_shape) == 3
        
        # dataset/dataloader

@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    ws = Workspace(cfg)
    
if __name__ == '__main__':
    main()