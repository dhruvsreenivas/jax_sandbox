import abc
import jax
from typing import Union
from dataset import Batch, OnPolicyBatch
from utils import MetricsDict


class Learner(abc.ABC):
    """Base learner class."""
    
    def act(self, x: jax.Array, eval: bool) -> jax.Array:
        """Base function for how to execute actions given a state, and optionally whether one is in eval mode or not."""
        
    def update(self, batch: Union[Batch, OnPolicyBatch], step: int) -> MetricsDict:
        """Base function for how to update a given learner on a batch and a step count."""