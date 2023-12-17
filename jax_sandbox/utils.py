import flax
from flax.training.train_state import TrainState
from flax import linen as nn
import jax
from typing import Mapping, Text

MetricsDict = Mapping[Text, jax.Array]

class TrainStateWithTarget(TrainState):
    """Train state with additional target params."""
    
    target_params: flax.core.FrozenDict