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

class Value_Network():

    def __init__(self, obs_space_dims):
        super().__init__()

        hidden_space1 = 16  # Nothing special with 16, feel free to change
        hidden_space2 = 8  # Nothing special with 32, feel free to change

        # Network
        self.value_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
            nn.Linear(hidden_space2, 1)
        )
        

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.value_net(x)
        return value


class Simulation:
    
    def __init__(self, render=False):
        
        self.env = gym.make("InvertedPendulum-v4")
        if render:
            self.env = gym.make("InvertedPendulum-v4", render_mode="human")
        self.obs_space_dims = self.env.observation_space.shape[0]
        self.action_space_dims = self.env.action_space.shape[0]
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.eps = 1e-6

        self.policy = Policy_Network(self.obs_space_dims, self.action_space_dims)
        self.pol_optimizer = torch.optim.AdamW(self.policy.policy_net.parameters(), lr=self.learning_rate)

        self.value = Value_Network(self.obs_space_dims)
        self.val_optimizer = torch.optim.AdamW(self.value.value_net.parameters(), lr=self.learning_rate)

        self.log_prob_buffer = []
        self.reward_buffer = []
        self.return_buffer = []
        self.episode_time_buffer = []
        self.episode_steps_buffer = []
        self.value_buffer = []
        self.td_buffer = []
        self.gae_buffer = []

        self.log_avg_reward = []
        self.log_avg_return = []
        self.log_avg_value = []

        self.training_step = 0
        self.eps_run = 0

        self.plot = True


    def sample_action(self, obs):
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        mean, dev = self.policy.forward(obs)
        distrib = Normal(mean, dev)
        action = distrib.sample()
        logp = distrib.log_prob(action).sum(axis=-1)
        self.log_prob_buffer.append(logp)
        return action
    
    def save_model(self, path):
        path = path + "/Cartpole_A2C.pth"
        torch.save(self.policy.policy_net.state_dict(), path)

    def save_value(self, path):
        path = path + "/Cartpole_A2C_val.pth"
        torch.save(self.value.value_net.state_dict(), path)
    
    def flush_post_ep(self):
        self.log_prob_buffer.clear()
        self.reward_buffer.clear()
        self.return_buffer.clear()
        self.value_buffer.clear()
        self.td_buffer.clear()
        self.gae_buffer.clear()

    def flush_post_iter(self):
        self.flush_post_ep()
        self.log_avg_return.clear()
        self.log_avg_reward.clear()
        self.episode_steps_buffer.clear()
        self.episode_time_buffer.clear()
        self.log_avg_value.clear()

    def flip_list(self, lt):
        length = len(lt)
        tmp_list = [0]*length
        for i, val in enumerate(lt):
            tmp_list[length-i-1] = val
        return tmp_list

    def get_return_buffer(self):
        rew = np.array(self.reward_buffer)
        rewards = np.flip(rew)
        gamma = self.gamma
        returns = []
        for i, reward in enumerate(rewards):
            if i == 0:
                returns.append(reward)
            else:
                returns.append(reward+gamma*returns[i-1])
        returns = self.flip_list(returns)
        self.return_buffer = list(returns)
        return self.return_buffer
    
    def policy_update(self):
        loss_pol = 0
        log_prob = self.log_prob_buffer
        for i in range(len(self.reward_buffer)):
            loss_pol+=self.log_prob_buffer[i]*(self.gae_buffer[i].detach())
        loss_pol*=-1
        self.pol_optimizer.zero_grad()
        loss_pol.backward()
        self.pol_optimizer.step()

    def value_update(self):
        loss_val = 0
        for i in range(len(self.return_buffer)):
            loss_val += (self.return_buffer[i]-self.value_buffer[i])**2
        self.val_optimizer.zero_grad()
        loss_val.backward()
        self.val_optimizer.step()

    def log_data(self):
        mean_rew = np.array(self.reward_buffer).mean()
        self.log_avg_reward.append(mean_rew)
        mean_ret = np.array(self.return_buffer).mean()
        self.log_avg_return.append(mean_ret)
        val_buffer = self.value_buffer
        for i, val in enumerate(range(len(val_buffer))):
            val_buffer[i] = val_buffer[i].detach().numpy()
        val_buffer = np.array(val_buffer)
        mean_val = val_buffer.mean()
        self.log_avg_value.append(mean_val)

    def moving_average(self, Y, n):
        Y_mva = []
        for index, y in enumerate(Y):
            if index>n:
                sum = 0
                for i in range(n):
                    sum+=Y[index-i]
                sum/=n
                Y_mva.append(sum)
            else:
                sum = 0
                for i in range(index+1):
                    sum+=Y[index-i]
                sum/=(index+1)
                Y_mva.append(sum)

        return Y_mva
    
    def get_td_buffer(self):
        values = np.array(self.value_buffer)
        rewards = np.array(self.reward_buffer)
        self.td_buffer = []
        for env in range(rewards.shape[1]):   
            env_td = []         
            for i, rew in enumerate(rewards[:, env]):
                print("i is: ", i)
                if i == rewards.shape[0]-1:
                    env_td.append(rew)
                else:
                    env_td.append(rew+values[i+1, env]-values[i, env])
            self.td_buffer.append(env_td)
        return self.td_buffer
    
    def get_gae_buffer(self, lmbda):
        gae = 0
        l = len(self.td_buffer)
        for i in range(len(self.td_buffer)):
            if i == 0:
                gae += self.td_buffer[l-i-1]
                self.gae_buffer.append(gae)
                continue
            gae = self.td_buffer[l-i-1] + self.gamma*lmbda*gae.detach()
            self.gae_buffer.append(gae)
        gae_buffer = [0]*l
        for i, gae in enumerate(self.gae_buffer):
            gae_buffer[l-i-1] = gae
        self.gae_buffer = gae_buffer
        return self.gae_buffer
    
    def load_weights(self, pol=None, val=None):
        if pol:
            self.policy.policy_net.load_state_dict(torch.load(pol))
        if val:
            self.value.value_net.load_state_dict(torch.load(val))


    def plot_training(self):
        fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True)
        X = range(self.eps_run)
        Y = self.log_avg_return
        Y = self.moving_average(Y, n=50)
        ax[0, 0].plot(X, Y)
        Y = self.log_avg_reward
        Y = self.moving_average(Y, n=50)
        ax[0, 1].plot(X, Y)
        Y = self.episode_steps_buffer
        Y = self.moving_average(Y, n=50)
        ax[0, 2].plot(X, Y)
        Y = np.cumsum(np.array(self.episode_steps_buffer))
        ax[1, 0].plot(X, Y)
        Y = np.cumsum(np.array(self.episode_time_buffer))
        ax[1, 1].plot(X, Y)        
        Y = self.log_avg_value
        ax[1, 2].plot(X, Y)        
        ax[0, 0].set_ylabel('Returns')
        ax[0, 0].set_ylim(bottom=0)
        ax[0, 1].set_ylabel('Rewards')
        ax[0, 2].set_ylabel("Episode Length")
        ax[1, 0].set_ylabel("# Steps")
        ax[1, 1].set_ylabel("Time (s)")
        ax[1, 2].set_ylabel("Average Value")
        plt.show()

    def train(self, num_eps, seed=1):
        
        train_time = time.time()

        for episode in range(num_eps):
            obs, info = self.env.reset(seed=seed)
            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32)
            self.value_buffer.append(self.value.forward(obs_tensor))
            done = False
            step_time = 0
            episode_start = time.time()
            num_steps = 0
            while not done:
                self.training_step += 1
                num_steps+=1
                action = self.sample_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32)
                val = self.value.forward(obs_tensor)
                self.value_buffer.append(val)
                self.reward_buffer.append(reward)
     
                step_dur = time.time()-episode_start
                episode_start = time.time()
                step_time+=step_dur

                done = terminated or truncated
            self.episode_steps_buffer.append(num_steps)
            step_time/=self.training_step
            self.eps_run+=1

            episode_time = time.time()
            episode_dur = episode_time - train_time
            train_time = time.time()
            self.episode_time_buffer.append(episode_dur)

            ## Update 
            self.get_return_buffer()
            self.get_td_buffer()
            self.get_gae_buffer(lmbda=0.99)
            self.policy_update()
            self.value_update()
            self.log_data()
            self.flush_post_ep()

            if (episode+1) % 100 == 0:
                avg_reward = self.log_avg_reward[-1]
                print("Episode:", episode, "Average Reward:", avg_reward, "Average Return: ", self.log_avg_return[-1])

            if (episode+1) % 1000 == 0 and self.plot:
                self.plot_training()
            


sim = Simulation()
pol = "/Users/stav.42/RL_Library/cartpole/weights/A2C.pth"
# sim.load_weights(pol=pol)
print("Simulation instantiated")

for seed in range(8):
    sim.train(num_eps=3000, seed=seed)
    sim.plot_training()
    # sim.save_model(path="/Users/stav.42/RL_Library/cartpole/weights/")
    # sim.save_value(path="/Users/stav.42/RL_Library/cartpole/weights/")
    # sim.flush_post_iter()
# sim.save_model(path="/Users/stav.42/RL_Library/cartpole/weights/")
# sim.save_value(path="/Users/stav.42/RL_Library/cartpole/weights/")
