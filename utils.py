import dill
import gym
import os.path as osp
from envs.atari import wrap_deepmind

def save(state, model_path: str):
    with open(model_path, 'wb') as f:
        dill.dump(state, f, protocol=2)
        
def load(model_path: str):
    try:
        with open(model_path, 'rb') as f:
            state = dill.load(f)
            return state
    except FileNotFoundError:
        print('cannot load model state.')
        exit()
        
ENVS = {
    # classic control
    'cartpole-v0': 'CartPole-v0',
    'cartpole-v2': 'CartPole-v2',
    'lunarlander-v0': 'LunarLander-v0',
    'lunarlander-v2': 'LunarLander-v2',
    
    # mujoco
    'halfcheetah-v0': 'HalfCheetah-v0',
    'halfcheetah-v2': 'HalfCheetah-v2',
    'halfcheetah-v3': 'HalfCheetah-v3',
    'hopper-v0': 'Hopper-v0',
    'hopper-v2': 'Hopper-v2',
    'hopper-v3': 'Hopper-v3',
    'walker2d-v0': 'Walker2d-v0',
    'walker2d-v2': 'Walker2d-v2',
    'walker2d-v3': 'Walker2d-v3',
    'ant-v0': 'Ant-v0',
    'ant-v2': 'Ant-v2',
    'ant-v3': 'Ant-v3',
    
    # atari
    'pong': 'PongNoFrameskip-v4',
    'breakout': 'BreakoutNoFrameskip-v4',
    'beamrider': 'BeamriderNoFrameskip-v4',
    'montezuma': 'MontezumaRevengeNoFrameskip-v4'
}

def is_discrete(env_name: str):
    if env_name in ['pong', 'breakout', 'beamrider', 'montezuma']:
        return True
    elif env_name.startswith('cartpole') or env_name.startswith('lunarlander'):
        return True
    return False

def make_gym_env(env_name: str, seed: int):
    real_env_name = ENVS[env_name]
    env = gym.make(real_env_name)
    env.seed(seed)
    
    if env_name in ['pong', 'breakout', 'beamrider', 'montezuma']:
        # do proper env wrapping
        expt_dir = './tmp/jax_sandbox/'
        env = gym.wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True, video_callable=False)
        env = wrap_deepmind(env)
        
    return env