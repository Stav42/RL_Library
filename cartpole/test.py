import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import torch.nn as nn
from torch.distributions.normal import Normal
import gymnasium as gym
    
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
    
class Simulation:
    
    def __init__(self, render=False):
        self.env = gym.make("InvertedPendulum-v4", render_mode="human")
        self.obs_space_dims = self.env.observation_space.shape[0]
        self.action_space_dims = self.env.action_space.shape[0]
        self.policy = Policy_Network(self.obs_space_dims, self.action_space_dims)

    def load_model(self, path):
        self.policy.policy_net.load_state_dict(torch.load(path))
    
    def sample_action(self, obs):
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        mean, dev = self.policy.forward(obs)
        distrib = Normal(mean, dev)
        action = distrib.sample()
        logp = distrib.log_prob(action).sum(axis=-1)
        return action
    
    def run(self, episodes):
        
        for episode in range(episodes):
            print("Episode Number: ", episode)
            obs, info = self.env.reset()
            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32)
            done = False
            while not done:
                action = self.sample_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

        
sim = Simulation()
print("Simulation instantiated")

path = "ppo/weights/Cartpole_PPO.pth"
sim.load_model(path)
sim.run(100)
        