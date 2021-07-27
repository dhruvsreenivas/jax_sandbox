import jax
from jax import grad, jit
import jax.numpy as jnp
import jax.nn as nn
from tensorflow_probability.substrates import jax as tfp

import torch
from torch.utils.data import TensorDataset, DataLoader

import optax
import haiku as hk
import numpy as np
import gym
import argparse


class BCLearner:
    '''
    BC learning agent for discrete environments. Not using rlax library just to see how well I do.
    '''

    def __init__(self, output_size, learning_rate):
        self.output_size = output_size

        self.network = hk.without_apply_rng(hk.transform(self.forward_pass))
        self.optimizer = optax.adam(learning_rate)

    def forward_pass(self, batch):
        states, _ = batch
        mlp = hk.Sequential([
            hk.Linear(64),
            nn.relu,
            hk.Linear(64),
            nn.relu,
            hk.Linear(self.output_size)
        ])

        logits = mlp(states)
        return logits

    @jit
    def loss(self, params, batch):
        # maximizing log probability of (s, a) batch
        states, actions = batch
        states = jnp.array(states.numpy())
        actions = jnp.array(actions.numpy())
        logits = self.network.apply(params, states)

        dist = tfp.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        return -jnp.mean(log_probs)

    @jit
    def update(self, params, opt_state, batch):
        grads = grad(self.loss)(params, batch)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='bc_jax')
    parser.add_argument(
        '--num_epochs', type=int, default=10000, help='number of training epochs to train BC agent (default 1e4)')
    parser.add_argument(
        '--eval-frequency', type=int, default=100, help='frequency when we log the progress of the agent'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64, help='batch size for training (default 64)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.01, help='learning rate for BC classifier (default 0.01)'
    )
    parser.add_argument(
        '--env', type=str, default='CartPole-v0', help='environment to run on'
    )
    parser.add_argument(
        '--expert-obs-path', type=str, default='cartpole_expert_observations.npy', help='path to expert observations dataset'
    )
    parser.add_argument(
        '--expert-acs-path', type=str, default='cartpole_expert_actions.npy', help='path to expert actions dataset'
    )

    args = parser.parse_args()

    # Env instantiation
    env = gym.make(args.env)

    if type(env.action_space) != gym.spaces.discrete.Discrete:
        raise Exception(
            'Current code only built for environments with discrete action spaces.')

    # BC learner initialization
    bc_learner = BCLearner(env.action_space.n, args.learning_rate)

    # network and optimizer parameter initialization
    rng_key = jax.random.PRNGKey(69)
    params = bc_learner.network.init(
        rng_key, jnp.zeros(env.observation_space.shape))
    opt_state = bc_learner.optimizer.init(params)

    # get observations and actions
    expert_obs = np.load(args.expert_obs_path)
    expert_acs = np.load(args.expert_acs_path)

    assert expert_obs.shape[0] == expert_acs.shape[0]

    # get dataloaders
    expert_dataset = TensorDataset(torch.from_numpy(
        expert_obs), torch.from_numpy(expert_acs))

    expert_dataloader = DataLoader(
        expert_dataset, batch_size=args.batch_size, shuffle=True)

    # training loop
    for i in range(args.num_epochs):
        for batch in expert_dataloader:
            params, opt_state = bc_learner.update(params, opt_state, batch)

        if i % args.eval_frequency == 0:
            loss = bc_learner.loss(params, batch)
            print(f'Loss after epoch {i}: {loss}')
