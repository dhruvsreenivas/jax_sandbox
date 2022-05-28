from common.neural_net import DiscreteQNetwork, ContinuousQNetwork
from common.utils import *
from common.dataset import TransitionBatch
import haiku as hk
import jax
import rlax

batch_q_learning_fn = jax.vmap(rlax.q_learning)

class DQN:
    def __init__(self, cfg):
        self.gamma = cfg.gamma
        
        assert not cfg.continuous, 'DQN only works with discrete action spaces.'
        
        # transform q net for init/apply (qnet is shell fn, so you can pass both online + target params to this)
        # q function should be deterministic (no randomness added)
        self.qnet = hk.without_apply_rng(hk.transform(lambda x: DiscreteQNetwork(cfg)(x)))
        
        # optimizer
        self.opt = get_opt_class(cfg.opt)(learning_rate=cfg.lr)
        if cfg.clip_grad_norm:
            self.opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), self.opt)
        
        # rng sequence
        self.rng_seq = hk.PRNGSequence(cfg.seed)
        
        # online + target params
        rng_key = next(self.rng_seq)
        self.online_params = self.target_params = self.qnet.init(rng_key, jnp.zeros((1, *cfg.obs_shape)))
        
        # opt state initialization
        self.opt_state = self.opt.init(self.online_params)
        
        # epsilon
        self.eps = 0.1 # TODO: don't hardcode
        
    def get_action(self, state):
        qs = self.qnet.apply(self.online_params, next(self.rng_seq), state)
        a = rlax.epsilon_greedy().sample(next(self.rng_seq), qs, self.eps)
        return a
        
    def learn(self, batch: TransitionBatch):
        # define loss fn, then take gradient and step
        def loss_fn(params, target_params, rng_key, batch):
            online_rng_key, target_rng_key = jax.random.split(rng_key, 2)
            targets = self.qnet.apply(target_params, target_rng_key, batch.next_states)
            outputs = self.qnet.apply(params, online_rng_key, batch.states)
            
            td_errors = batch_q_learning_fn(
                outputs,
                batch.actions,
                batch.rewards,
                self.gamma * jnp.ones_like(batch.rewards),
                targets
            )
            
            losses = rlax.l2_loss(td_errors)
            return jnp.mean(losses)

        rng_key = next(self.rng_seq)
        loss = loss_fn(self.online_params, self.target_params, rng_key, batch)
        grads = jax.grad(loss_fn)(self.online_params, self.target_params, rng_key, batch)
        updates, new_opt_state = self.opt.update(grads, self.opt_state)
        new_params = optax.apply_updates(self.online_params, updates)
        
        self.online_params = new_params
        self.opt_state = new_opt_state
        
        return loss.item()