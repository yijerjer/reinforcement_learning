import numpy as np
import torch
from torch.distributions import Categorical
from torch.optim import Adam
from algorithms.mlp import MLP


class ActorCritic:
    def __init__(
        self, env, optim=Adam, policy_lr=0.001, value_lr=0.001,
        policy_hidden_size=[32], value_hidden_size=[32], gamma=0.9,
        batch_size=5000, epochs=50, update_every=50, render=False
    ):
        self.env = env
        self.batch_size = batch_size
        self.render = render
        self.epochs = epochs
        self.gamma = gamma
        self.update_every = update_every

        obs_size = np.prod(env.observation_space.shape)
        action_size = env.action_space.n
        self.policy_mlp = MLP([obs_size] + policy_hidden_size + [action_size])
        self.policy_optim = optim(self.policy_mlp.parameters(), lr=policy_lr)
        self.value_mlp = MLP([obs_size] + value_hidden_size + [1])
        self.value_optim = optim(self.value_mlp.parameters(), lr=value_lr)

    def train(self):
        for epoch in range(self.epochs):
            render = False
            if self.render:
                render = True if epoch % 5 == 0 else False

            returns, lens = self.train_single_batch(render=render)
            print("Epoch %2d, Return: %5.1f, Length: %3d" %
                  (epoch, np.mean(returns), np.mean(lens)))

    def train_single_batch(self, render=False):
        group_data = []
        batch_returns = []
        batch_lens = []
        episode_rewards = []

        done = False
        obs = self.env.reset()
        I_val = 1

        first_episode_render = True
        for t in range(self.batch_size):
            if render and first_episode_render:
                self.env.render()

            curr_obs = obs
            action = self.get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, reward, done, _ = self.env.step(action)
            episode_rewards.append(reward)
            group_data.append((curr_obs, action, reward, obs, done, I_val))

            I_val *= self.gamma

            if t > 0 and t % self.update_every == 0:
                for data in group_data:
                    self.update(data)
                group_data = []

            if done:
                ep_return, ep_len = sum(episode_rewards), len(episode_rewards)
                batch_returns.append(ep_return)
                batch_lens.append(ep_len)

                episode_rewards = []
                obs, done = self.env.reset(), False
                I_val = 1

                first_episode_render = False

        return batch_returns, batch_lens

    def update(self, data):
        obs, action, reward, next_obs, done, I_val = data
        obs = torch.as_tensor([obs], dtype=torch.float32)
        next_obs = torch.as_tensor([next_obs], dtype=torch.float32)
        action = torch.as_tensor([action], dtype=torch.float32)
        reward = torch.as_tensor(reward, dtype=torch.float32)

        error = self.get_value_error(obs, next_obs, reward, done)

        self.value_optim.zero_grad()
        value_loss = self.value_update(obs, error)
        value_loss.backward()
        self.value_optim.step()

        self.policy_optim.zero_grad()
        policy_loss = self.policy_update(obs, action, error, I_val)
        policy_loss.backward()
        self.policy_optim.step()

    def policy(self, obs):
        mlp_out = self.policy_mlp(obs)
        return Categorical(logits=mlp_out)

    def get_action(self, obs):
        policy_dist = self.policy(obs)
        action = policy_dist.sample().item()
        return action

    def policy_update(self, obs, action, error, I):
        policy_dist = self.policy(obs)
        log_proba = policy_dist.log_prob(action)
        print(-(error * I * log_proba))
        return -(error * I * log_proba)

    def state_value(self, obs):
        mlp_out = self.value_mlp(obs)
        return mlp_out

    def value_update(self, obs, error):
        value = self.state_value(obs)
        return -(error * value)

    def get_value_error(self, obs, next_obs, reward, done):
        value = self.state_value(obs).clone().detach()
        next_value = 0 if done else self.state_value(next_obs).clone().detach()

        return (reward + self.gamma * next_value - value)
