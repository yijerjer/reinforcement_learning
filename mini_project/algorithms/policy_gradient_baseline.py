import numpy as np
import torch
from torch.distributions import Categorical
from torch.optim import Adam
from algorithms.mlp import MLP


class PolicyGradientBaseline:
    def __init__(
        self, env, optim=Adam, policy_lr=0.01, value_lr=0.1,
        policy_hidden_size=[32], value_hidden_size=[32],
        batch_size=5000, render=False
    ):
        self.env = env
        self.batch_size = batch_size
        self.render = render

        obs_size = np.prod(env.observation_space.shape)
        action_size = env.action_space.n
        self.policy_mlp = MLP([obs_size] + policy_hidden_size + [action_size])
        self.policy_optim = optim(self.policy_mlp.parameters(), lr=policy_lr)
        self.value_mlp = MLP([obs_size] + value_hidden_size + [1])
        self.value_optim = optim(self.value_mlp.parameters(), lr=value_lr)

    def train(self):
        for epoch in range(50):
            render = False
            if self.render:
                render = True if epoch % 5 == 0 else False

            loss, returns, lens = self.train_single_batch(render=render)
            print("Epoch %2d, Loss %5.1f, Return: %5.1f, Length: %3d" %
                  (epoch, loss.item(), np.mean(returns), np.mean(lens)))

    def train_single_batch(self, render=False):
        timestep = 0

        batch_obss = []
        batch_actions = []
        batch_weights = []
        batch_returns = []
        batch_lens = []
        episode_rewards = []

        done = False
        obs = self.env.reset()

        first_episode_render = True
        while True:

            if render and first_episode_render:
                self.env.render()

            batch_obss.append(obs)
            action = self.get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, reward, done, _ = self.env.step(action)
            batch_actions.append(action)
            episode_rewards.append(reward)

            timestep += 1
            if done:
                episode_return = sum(episode_rewards)
                episode_len = len(episode_rewards)
                batch_returns.append(episode_return)
                batch_lens.append(episode_len)

                batch_weights += [sum(episode_rewards[i:]) for i, _ in
                                  enumerate(episode_rewards)]

                first_episode_render = False
                obs, done, episode_rewards = self.env.reset(), False, []

                if len(batch_obss) > self.batch_size:
                    break

        obss = torch.as_tensor(batch_obss, dtype=torch.float32)
        actions = torch.as_tensor(batch_actions, dtype=torch.float32)
        weights = torch.as_tensor(batch_weights, dtype=torch.float32)

        self.policy_optim.zero_grad()
        policy_updates = self.policy_update(obss, actions, weights)
        policy_updates.backward()
        self.policy_optim.step()

        self.value_optim.zero_grad()
        value_updates = self.value_update(obss, weights)
        value_updates.backward()
        self.value_optim.step()

        return policy_updates, batch_returns, batch_lens

    def policy(self, obs):
        mlp_out = self.policy_mlp(obs)
        return Categorical(logits=mlp_out)

    def get_action(self, obs):
        policy_dist = self.policy(obs)
        action = policy_dist.sample().item()
        return action

    def policy_update(self, obss, actions, returns):
        policy_dist = self.policy(obss)
        log_proba = policy_dist.log_prob(actions)
        value_errors = self.get_value_error(obss, returns)
        return -(value_errors * log_proba).mean()

    def state_value(self, obs):
        mlp_out = self.value_mlp(obs)
        return mlp_out

    def value_update(self, obss, returns):
        value = self.state_value(obss)
        value_errors = self.get_value_error(obss, returns)
        return -(value_errors * value).mean()

    def get_value_error(self, obss, returns):
        return (returns - self.state_value(obss)).clone().detach()
