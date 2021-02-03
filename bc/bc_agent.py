import jax
from jax import jit, vmap
import jax.numpy as jnp
import jax.nn as nn
import optax
import rlax
import haiku as hk
import torch
from torch.utils.data import DataLoader
import numpy as np
import gym
import argparse


class BC_ConvNet(hk.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv1 = hk.Conv2D(32, kernel_shape=8, stride=4)
        self.relu = nn.relu
        self.conv2 = hk.Conv2D(64, kernel_shape=4, stride=2)
        self.conv3 = hk.Conv2D(64, kernel_shape=3, stride=1)
        self.flatten = hk.Flatten()
        self.linear = hk.Linear(output_size)
        self.softmax = nn.softmax

    def __call__(self, input_img):
        input_img = input_img.astype(jnp.float32)
        x = self.conv1(input_img)
        x = self.relu(x)
        x = self.conv2(input_img)
        x = self.relu(x)
        x = self.conv3(input_img)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        logits = self.softmax(x)
        return logits


# implements same BC agent that the people in DRIL made
rng_seq = hk.PRNGSequence(42)


def cross_entropy_loss(outputs, actions):
    '''
    Calculates cross entropy loss (-sum(actions * log(outputs)))
    '''
    l = len(actions[0])
    for i in range(len(actions)):
        # one hot actions
        actions[i] = nn.one_hot(actions[i], l)
    return -1 * jnp.sum(actions * jnp.log(outputs))


class BCAgent:
    def __init__(self, input_size, num_actions, learning_rate=0.01, normalize=False):
        self.input_size = input_size
        self.normalize = normalize

        convnet = BC_ConvNet(num_actions)
        self.network = hk.without_apply_rng(hk.transform(convnet))

        self.optimizer = optax.adam(learning_rate)

    def initial_params(self):
        rand_input = jax.random.normal(self.input_size)
        params = self.network.init(next(rng_seq), rand_input)
        return params

    def loss_fn(self, params, rng_key, obs, acs):
        # outputs should have shape (num_actions)
        outputs = self.network.apply(params, rng_key, obs)
        cross_ent_loss = vmap(cross_entropy_loss(
            outputs, acs), in_axes=(0, None))
        return cross_ent_loss

    # update with ADAM
    @jit
    def update(self, params, opt_state, obs, acs):
        grads = jax.grad(self.loss_fn)(params, obs, acs)
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='bc_jax')
    parser.add_argument(
        '--num_training_epochs', type=int, default=100000, help='number of training epochs to train BC agent (default 1e5)')
    parser.add_argument(
        '--log-interval', type=int, default=10, help='interval when we log the progress of the agent'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64, help='batch size for training (default 64)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.01, help='learning rate for BC classifier (default 0.01)'
    )
    parser.add_argument(
        '--normalize', action='store_true', default=False, help='flag for if we normalize'
    )
    parser.add_argument(
        '--env', type=str, default='PongNoFrameskip-v4', help='environment to run on'
    )
    parser.add_argument(
        '--expert-obs-path', type=str, default='pong_expert_observations.npy', help='path to expert observations dataset'
    )
    parser.add_argument(
        '--expert-acs-path', type=str, default='pong_expert_actions.npy', help='path to expert actions dataset'
    )

    args = parser.parse_args()

    expert_observations = jnp.array(np.load(args.expert_obs_path))
    expert_actions = jnp.array(np.load(args.expert_acs_path))

    input_size = (4, 84, 84)
    num_actions = gym.make('PongNoFrameskip-v4').action_space.n

    bc_agent = BCAgent(input_size, num_actions,
                       learning_rate=args.learning_rate, normalize=args.normalize)

    # initialize agent's parameters
    params = bc_agent.initial_params()
    opt_state = bc_agent.optimizer.init(params)

    for epoch in range(args.num_training_epochs):
        for b in range(len(expert_observations) // args.batch_size):
            if b == len(expert_observations) // args.batch_size - 1:
                batch_obs = expert_observations[b * batch_size:]
                batch_acs = expert_actions[b * batch_size:]
            else:
                batch_obs = expert_observations[b *
                                                batch_size: (b+1) * batch_size]
                batch_acs = expert_actions[b * batch_size: (b+1) * batch_size]
            params, opt_state = bc_agent.update(
                params, opt_state, batch_obs, batch_acs)

            if epoch % args.log_interval == 0:
                l = bc_agent.loss_fn(params, rng_key, batch_obs, batch_acs)
                print(f'current loss: {l}')
