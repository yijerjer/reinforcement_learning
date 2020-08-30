import numpy as np
import torch
from torch.distributions import Categorical
from torch.optim import Adam
from mlp import MLP


class PolicyGradient:
    def __init__(
        self, env, optim=Adam, lr=0.01, hidden_size=[64],
        batch_size=5000, n_episodes=2000, render=False
    ):
        self.env = env
        self.batch_size = batch_size
        self.n_episodes = n_episodes
        self.lr = lr
        self.render = render

        obs_size = np.prod(env.observation_space.shape)
        action_size = env.action_space.n
        self.mlp = MLP([obs_size] + hidden_size + [action_size])
        self.optim = optim(self.mlp.parameters(), lr=lr)

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

        self.optim.zero_grad()
        batch_loss = self.policy_update(
            torch.as_tensor(batch_obss, dtype=torch.float32),
            torch.as_tensor(batch_actions, dtype=torch.float32),
            torch.as_tensor(batch_weights, dtype=torch.float32)
        )
        batch_loss.backward()
        self.optim.step()

        return batch_loss, batch_returns, batch_lens

    def policy(self, obs):
        mlp_out = self.mlp(obs)
        return Categorical(logits=mlp_out)

    def get_action(self, obs):
        policy_dist = self.policy(obs)
        action = policy_dist.sample().item()
        return action

    def policy_update(self, obs, actions, returns):
        policy_dist = self.policy(obs)
        log_proba = policy_dist.log_prob(actions)
        return -(returns * log_proba).mean()
