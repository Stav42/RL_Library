import random
import numpy as np
import torch.nn as nn
from torch.distributions.normal import Normal
import gymnasium as gym
import torch

class Policy_Network():

    def __init__(self, obs_space_dims, action_space_dims):
        super().__init__()

        hidden_space1 = 16  # Nothing special with 16, feel free to change
        hidden_space2 = 32  # Nothing special with 32, feel free to change

        # Network
        self.policy_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
            nn.Linear(hidden_space2, action_space_dims)
        )

        log_std = -0.5 * np.ones(action_space_dims, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_mean = self.policy_net(x)
        std = torch.exp(self.log_std)   

        return action_mean, std


def sample_action(nn, obs, action=None):
    obs = torch.tensor(np.array(obs), dtype=torch.float32)
    mean, dev = nn.forward(obs)
    distrib = Normal(mean, dev)
    if action is None:
        action = distrib.sample()
    logp = distrib.log_prob(action).sum(axis=-1)
    return action, logp

nn = Policy_Network(3, 1)

action, logp = sample_action(nn, [3, 4, 5])
print(action)
print(logp)

with torch.no_grad():
    action, logp = sample_action(nn, [3, 4, 5])
    print(action)
    print(logp)

print(action)
print(logp)