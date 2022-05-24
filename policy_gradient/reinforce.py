import jax
import haiku as hk
from common.neural_net import *
from common.dataset import TransitionBatch
from common.utils import get_opt_class
import optax

class REINFORCE:
    '''Continuous REINFORCE algorithm.'''
    def __init__(self, cfg):
        policy_fn = Policy(cfg)
        self.policy = hk.transform(lambda x: policy_fn(x))
        
        self.opt = get_opt_class(cfg.opt)(learning_rate=cfg.lr)
        self.opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), self.opt)
        
        # rng sequence
        self.rng_seq = hk.PRNGSequence(cfg.seed)
        
        # initialization of params and opt state
        self.params = self.policy.init(next(self.rng_seq), jnp.zeros(1, *cfg.obs_shape))
        self.opt_state = self.opt.init(self.params)
        
    def learn(self, batch: TransitionBatch) -> None:
        # define loss fn and apply updates w.r.t parameters
        def loss_fn(params, rng_key, batch):
            sum_rewards = jnp.sum(batch.rewards) # R(traj)
            
            action_dists = self.policy.apply(params, rng_key, batch.states)
            traj_lp = action_dists.log_prob(batch.actions).sum() # log pi_\theta(traj)
            
            return sum_rewards * traj_lp
        
        loss = loss_fn(self.params, next(self.rng_seq), batch)
        grads = jax.grad(loss_fn)(self.params, batch)
        updates, new_opt_state = self.opt.update(grads, self.opt_state)
        new_params = optax.apply_updates(self.params, updates)
        
        self.params = new_params
        self.opt_state = new_opt_state
        
        return loss.item()
            
            