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
        
        self.env = gym.make("InvertedPendulum-v4")
        if render:
            self.env = gym.make("InvertedPendulum-v4", render_mode="human")
        self.obs_space_dims = self.env.observation_space.shape[0]
        self.action_space_dims = self.env.action_space.shape[0]
        self.learning_rate = 1e-4
        self.gamma = 0.9
        self.eps = 1e-6

        self.policy = Policy_Network(self.obs_space_dims, self.action_space_dims)
        self.pol_optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

        self.log_prob_buffer = []
        self.reward_buffer = []
        self.return_buffer = []

        self.log_avg_reward = []
        self.log_avg_return = []

        self.training_step = 0

        self.plot = True


    def sample_action(self, obs):
        obs = torch.tensor(np.array(obs))
        mean, dev = self.policy(obs)
        distrib = Normal(mean, dev)
        action = distrib.sample()
        logp = distrib.log_prob(action).sum(axis=-1)
        self.log_prob_buffer.append(logp)
        return action
    
    def flush_post_ep(self):
        self.log_prob_buffer.clear()
        self.reward_buffer.clear()
        self.return_buffer.clear()

    def flush_post_iter(self):
        self.flush_post_ep()
        self.log_avg_return.clear()
        self.log_avg_reward.clear()

    def get_return_buffer(self):
        rewards = np.flip(np.array(self.reward_buffer))
        acc_rew = 0
        gamma = self.gamma
        returns = []
        for i, reward in enumerate(rewards):
            acc_rew += (gamma**i) * reward
        for i, reward in enumerate(rewards):
            if i == 0:
                returns.append(acc_rew)
                prev_reward = reward
            else:
                returns.append((returns[-1]-prev_reward)/gamma)
                prev_reward = reward
        self.return_buffer = list(returns)
        return returns
    
    def update(self):
        loss = 0
        log_prob = torch.tensor(np.array(self.log_prob_buffer))
        for i in range(len(self.reward_buffer)):
            loss+=log_prob*self.return_buffer[i]
        loss*=-1
        self.pol_optimizer.zero_grad()
        loss.backward()
        self.pol_optimizer.step()

    def log_data(self):
        mean_rew = np.array(self.reward_buffer).mean()
        self.log_avg_reward.append(mean_rew)
        mean_ret = np.array(self.return_buffer).mean()
        self.log_avg_return.append(mean_ret)

    def plot_training(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True)
        X = range(self.training_step)
        Y = self.log_avg_return
        ax[0, 0].plot(X, Y)
        Y = self.log_avg_reward
        ax[0, 1].plot(X, Y)
        ax[0, 0].set_ylabel('Returns')
        ax[0, 1].set_ylabel('Rewards')
        plt.show()
        



    def train(self, num_eps, seed=1):
        
        for episode in range(num_eps):
            self.training_step += 1
            obs, info = self.env.reset(seed=seed)
            done = False
            while not done:
                action = self.sample_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.reward_buffer.append(reward)

                done = terminated or truncated

            ## Update 
            self.get_return_buffer()
            self.update()
            self.log_data()
            self.flush_post_ep()

            if episode % 100 == 0:
                avg_reward = self.log_avg_reward[-1]
                print("Episode:", episode, "Average Reward:", avg_reward, "Average Return: ", self.log_avg_return[-1])

            if episode % 1000 == 0 and self.plot:
                self.plot_training()
            

def main():

    sim = Simulation()
    for seed in range(4):
        sim.train(num_eps=3000, seed=seed)
        sim.plot_training()
        sim.flush_post_iter()