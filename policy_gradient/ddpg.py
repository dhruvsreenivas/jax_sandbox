import jax
import jax.numpy as jnp
import haiku as hk
import optax
import rlax
from common.nets import *
from common.dataset import TransitionBatch
from common.utils import opt_class

class DDPG:
    def __init__(self, cfg):
        pass