from collections import namedtuple
from collections import deque
import numpy as np

TransitionBatch = namedtuple('Transition', ['states', 'actions', 'rewards', 'next_states', 'dones'])
ExpertBatch = namedtuple('ExpertBatch', ['states', 'actions'])

class ExpertDataset:
    '''Fixed size dataset.'''
    def __init__(self, states, actions):
        # states, actions both numpy arrays (for CPU usage, not to overload GPU probably)
        self.states = states
        self.actions = actions
    
    def sample(self, batch_size):
        idxes = np.random.randint(len(self.states), size=batch_size)
        return ExpertBatch(states=self.states[idxes], actions=self.actions[idxes])
        
class ReplayBuffer:
    def __init__(self, maxsize=1000000):
        self.states = deque([], maxlen=maxsize)
        self.actions = deque([], maxlen=maxsize)
        self.rewards = deque([], maxlen=maxsize)
        self.next_states = deque([], maxlen=maxsize)
        self.dones = deque([], maxlen=maxsize)
        
    def add(self, experience):
        self.states.extend(experience.states)
        self.actions.extend(experience.actions)
        self.rewards.extend(experience.rewards)
        self.next_states.extend(experience.next_states)
        self.dones.extend(experience.dones)
        
    def sample(self, batch_size):
        idxes = np.random.randint(len(self.states), size=batch_size)
        states = np.concatenate([self.states[idx] for idx in idxes])
        actions = np.concatenate([self.actions[idx] for idx in idxes])
        rewards = np.concatenate([self.rewards[idx] for idx in idxes])
        next_states = np.concatenate([self.next_states[idx] for idx in idxes])
        dones = np.concatenate([self.dones[idx] for idx in idxes])
        
        return TransitionBatch(states, actions, rewards, next_states, dones)
        
        