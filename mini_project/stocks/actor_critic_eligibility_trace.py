import numpy as np
import torch
from torch.optim import Adam
from mlp import MLP, CategoricalMLP
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


class ActorCriticEligibilityTrace:
    def __init__(
        self, env, optim=Adam, policy_lr=0.001, value_lr=0.001,
        policy_hidden_size=[32], value_hidden_size=[32], gamma=0.95,
        policy_lambda=0.9, value_lambda=0.9,
        batch_size=3000, epochs=15, update_every=50, render=False
    ):
        self.env = env
        self.batch_size = batch_size
        self.render = render
        self.epochs = epochs
        self.gamma = gamma
        self.policy_lambda = policy_lambda
        self.value_lambda = value_lambda
        self.update_every = update_every
        self.writer_count = 0

        obs_size = env.obs_space_size
        action_size = env.action_space_size
        self.policy_mlp = CategoricalMLP([obs_size] + policy_hidden_size
                                         + [action_size])
        self.policy_optim = optim(self.policy_mlp.parameters(), lr=policy_lr)
        self.value_mlp = MLP([obs_size] + value_hidden_size + [1])
        self.value_optim = optim(self.value_mlp.parameters(), lr=value_lr)

    def train(self):
        for epoch in range(self.epochs):
            returns, lens = self.train_single_batch(render=self.render)
            print("Epoch %2d, Return: %5.1f, Length: %3d" %
                  (epoch, np.mean(returns), np.mean(lens)))

    def train_single_batch(self, render=False):
        batch_returns = []
        batch_lens = []
        episode_rewards = []

        done = False
        obs = self.env.reset()
        I_val = 1
        self.policy_trace = self.create_trace(self.policy_mlp)
        self.value_trace = self.create_trace(self.value_mlp)

        for t in range(self.batch_size):
            curr_obs = obs
            action, log_prob = self.policy_mlp(
                torch.as_tensor(obs, dtype=torch.float32)
            )
            obs, reward, done, _ = self.env.step(action.detach().numpy())
            episode_rewards.append(reward)
            error = self.update((curr_obs, action, log_prob, reward, obs, done, I_val))

            I_val *= self.gamma

            if done:
                ep_return, ep_len = sum(episode_rewards), len(episode_rewards)
                batch_returns.append(ep_return)
                batch_lens.append(ep_len)

                episode_rewards = []
                obs, done = self.env.reset(), False
                I_val = 1

                writer.add_scalar("Returns", ep_return, self.writer_count)
                writer.add_scalar("Error", error, self.writer_count)
                self.writer_count += 1

        self.save_mlps()
        return batch_returns, batch_lens

    def update(self, data):
        obs, action, log_prob, reward, next_obs, done, I_val = data
        obs = torch.as_tensor([obs], dtype=torch.float32)
        next_obs = torch.as_tensor([next_obs], dtype=torch.float32)
        action = torch.as_tensor(action, dtype=torch.float32)
        reward = torch.as_tensor(reward, dtype=torch.float32)

        error = self.get_value_error(obs, next_obs, reward, done)

        self.value_optim.zero_grad()
        self.value_set_grad(obs, error)
        self.value_optim.step()

        self.policy_optim.zero_grad()
        self.policy_set_grad(obs, action, log_prob, error, I_val)
        self.policy_optim.step()

        return error

    def policy_set_grad(self, obs, action, log_prob, error, I):
        log_prob.backward()
        for i, p in enumerate(self.policy_mlp.parameters()):
            self.policy_trace[i] = (
                self.gamma * self.policy_lambda * self.policy_trace[i]
                + I * p.grad
            )
            p.grad = -(error * self.policy_trace[i])

    def state_value(self, obs):
        mlp_out = self.value_mlp(obs)
        return mlp_out

    def value_set_grad(self, obs, error):
        value = self.state_value(obs)
        value.backward()
        for i, p in enumerate(self.value_mlp.parameters()):
            self.value_trace[i] = (
                self.gamma * self.value_lambda * self.value_trace[i]
                + p.grad
            )
            p.grad = -(error * self.value_trace[i])

    def get_value_error(self, obs, next_obs, reward, done):
        value = self.state_value(obs).clone().detach()
        next_value = 0 if done else self.state_value(next_obs).clone().detach()

        return (reward + self.gamma * next_value - value).item()

    def create_trace(self, model):
        trace = []
        for p in model.parameters():
            trace.append(torch.zeros(p.shape))
        return trace

    def save_mlps(self):
        torch.save(self.policy_mlp.state_dict(), "policy_mlp.pth")
        torch.save(self.value_mlp.state_dict(), "value_mlp.pth")
