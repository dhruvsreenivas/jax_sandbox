from jax import grad, jit, vmap
import jax.numpy as jnp
import jax.nn as nn

import optax
import rlax
import haiku as hk

batch_q_learning = vmap(rlax.q_learning)


class DQNLearner:
    def __init__(self, obs_dim, num_actions, gamma=0.99):
        self.input_dim = obs_dim
        self.output_dim = num_actions

        self.disc_factor = gamma

        self.q_network = hk.transform(self.forward_pass)

    def forward_pass(self, batch):
        states, _, _, _, _ = batch

        mlp = hk.Sequential([
            hk.Linear(64),
            nn.relu,
            hk.Linear(64),
            nn.relu,
            hk.Linear(self.output_size)
        ])

        q_vals = mlp(states)
        return q_vals

    @jit
    def loss(self, params, batch):
        states, actions, rewards, next_states, _ = batch

        q_vals = self.q_network.apply(params, states)

        one_step_q_vals = self.q_network.apply(params, next_states)

        td_errors = batch_q_learning(
            q_vals, actions, rewards, self.discount_factor, one_step_q_vals)

        return jnp.mean(rlax.l2_loss(td_errors))

    @jit
    def update(self, params, opt_state, batch, rng_key):
        grads = grad(self.loss)(params, rng_key, batch)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state
