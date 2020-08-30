import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical


class MLP(nn.Module):
    def __init__(self, sizes, activation_func=nn.ReLU):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            act_func = activation_func if i < (len(sizes) - 2) else nn.Identity
            layers += [
                nn.Linear(sizes[i], sizes[i+1]), act_func()
            ]
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        output = self.sequential(x)
        return output


class CategoricalMLP(nn.Module):
    def __init__(self, sizes, activation_func=nn.ReLU):
        super(CategoricalMLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            act_func = activation_func if i < (len(sizes) - 2) else nn.Identity
            layers += [
                nn.Linear(sizes[i], sizes[i+1]), act_func()
            ]
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        output = self.sequential(x)
        dist = Categorical(logits=output)

        sample = dist.sample()
        log_proba = dist.log_prob(sample)

        return sample, log_proba


class GaussianMLP(nn.Module):
    def __init__(self, sizes, action_limit, activation_func=nn.ReLU):
        super(GaussianMLP, self).__init__()
        self.action_limit = action_limit

        layers = []
        for i in range(len(sizes) - 2):
            layers += [nn.Linear(sizes[i], sizes[i+1]), nn.ReLU()]
        self.core_net = nn.Sequential(*layers)
        self.mean_net = nn.Linear(sizes[-2], sizes[-1])
        self.log_std_net = nn.Linear(sizes[-2], sizes[-1])

    def forward(self, x):
        core_out = self.core_net(x)
        mean = self.mean_net(core_out)
        log_std = self.log_std_net(core_out)
        std = torch.exp(log_std)

        norm_dist = Normal(mean, std)
        sample = norm_dist.rsample()

        log_proba = norm_dist.log_prob(sample).sum()
        sample = torch.tanh(sample) * self.action_limit

        return sample, log_proba
