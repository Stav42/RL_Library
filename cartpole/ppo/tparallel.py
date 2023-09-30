import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import time
import torch.nn as nn
from torch.distributions.normal import Normal
import gymnasium as gym
from distutils.util import strtobool
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from typing import Optional
import wandb
from torch.utils.tensorboard import SummaryWriter

import functools

from datetime import date
today = date.today()
d = str(today.strftime("%b-%d-%Y"))

def parse_args():
    parser =  argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=d, help="name of this experiment")
    parser.add_argument('--gym-id', type=str, default="InvertedPendulum-v4", help="id of this environment")
    parser.add_argument('--learning-rate', type=float, default=2.5e-4, help="set learning rate for algorithm")
    parser.add_argument('--seed', type=int, default=1, help="seed of this experiment")
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs="?", const="True", help="To enable WandB tracking")
    parser.add_argument('--wandb-project-name', type=str, default="Prototype", help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument('--num_minibatches', type=int, default=32, help="the number of mini-batches")
    parser.add_argument('--num_envs', type=int, default=8, help="number of environments to activate")
    parser.add_argument('--total_timesteps', type=int, default=100000, help="number of global steps to execute")
    parser.add_argument('--batch_size', type=int, default=8, help="number of environments to activate")
    parser.add_argument('--num_steps', type=int, default=1600, help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args



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
    
def make_env(gym_id, seed, rank, capture_video=None, run_name=None):
    def _init():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.reset(seed=seed+rank)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return _init


class Simulation:
    
    def __init__(self, render=False):

        self.env_id = "InvertedPendulum-v4"
        self.num_cpu = 16
        self.envs = SubprocVecEnv([make_env(args.gym_id, seed=i, rank=i) for i in range(args.num_envs)])
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.eps = 1e-6

        self.obs_space_dim = self.envs.observation_space.shape[0]
        self.action_space_dim = self.envs.action_space.shape[0]

        self.policy = Policy_Network(self.obs_space_dim, self.action_space_dim)
        self.pol_optimizer = torch.optim.AdamW(self.policy.policy_net.parameters(), lr=self.learning_rate)

        self.value = Value_Network(self.obs_space_dim)
        self.val_optimizer = torch.optim.AdamW(self.value.value_net.parameters(), lr=self.learning_rate)

        self.log_prob_buffer = []
        self.log_prob_buffer = torch.zeros((args.num_steps, args.num_envs))
        self.reward_buffer = torch.zeros((args.num_steps, args.num_envs))
        self.return_buffer = torch.zeros((args.num_steps, args.num_envs))
        self.update_time_buffer = []
        self.update_steps_buffer = []
        self.value_buffer = torch.zeros((args.num_steps, args.num_envs))
        self.td_buffer = torch.zeros((args.num_steps, args.num_envs))
        self.gae_buffer = torch.zeros((args.num_steps, args.num_envs))
        self.epsilon = 0

        self.log_avg_reward = []
        self.log_avg_return = []
        self.log_avg_value = []

        self.old_log_prob = torch.zeros((args.num_steps, args.num_envs))
        self.training_step = 0
        self.eps_run = 0

        self.clip_coeff = 0.2
        self.plot = True

        self.writer = None


    def tensorboard_init(self):
        run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.writer.add_text(
            "Hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
        )

    
    def wandb_init(self):
        run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        wandb.init(
            project=args.wandb_project_name, 
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )


    def sample_action(self, obs, step):
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        mean, dev = self.policy.forward(obs)
        distrib = Normal(mean, dev)
        action = distrib.sample()
        logp = distrib.log_prob(action).sum(axis=-1)
        self.log_prob_buffer[step, :] = logp
        return action
    
    def save_model(self, path):
        path = path + "/Cartpole_PPO.pth"
        torch.save(self.policy.policy_net.state_dict(), path)

    def save_value(self, path):
        path = path + "/Cartpole_PPO_val.pth"
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
        self.update_steps_buffer.clear()
        self.update_time_buffer.clear()
        self.log_avg_value.clear()

    def flip_list(self, lt):
        length = len(lt)
        tmp_list = [0]*length
        for i, val in enumerate(lt):
            tmp_list[length-i-1] = val
        return tmp_list

    def get_return_buffer(self):
        rewards = torch.flip(self.reward_buffer, [0])
        for env in range(rewards.size()[1]):
            gamma = self.gamma
            gamma = 1
            for i, reward in enumerate(rewards[:, env]):
                if i == 0:
                    self.return_buffer[i, env] = reward
                else:
                    self.return_buffer[i, env] = reward + self.return_buffer[i-1, env]*gamma
        self.return_buffer = torch.flip(self.return_buffer, [0])
        return self.return_buffer
    
    def get_gae_buffer(self, lmbda):
        gamma = self.gamma
        for env in range(self.td_buffer.size()[1]):
            l = self.td_buffer.size()[0]
            for i in range(l):
                if i == 0:
                    self.gae_buffer[l-i-1][env] = self.td_buffer[l-i-1][env]
                else:
                    self.gae_buffer[l-i-1][env] = self.td_buffer[l-i-1][env] + lmbda*gamma*self.gae_buffer[l-i][env]
        return self.gae_buffer

    def get_td_buffer(self):
        for env in range(self.reward_buffer.size()[1]):   
            for i, rew in enumerate(self.reward_buffer[:, env]):
                if i == self.reward_buffer.size()[0]-1:
                    self.td_buffer[i, env] = rew
                else:
                    self.td_buffer[i, env] = rew + self.value_buffer[i+1, env] - self.value_buffer[i, env]
        return self.td_buffer
    
    def policy_update(self):
        loss_pol = 0
        log_prob = self.log_prob_buffer
        for env in range(args.num_envs):
            for i in range(self.reward_buffer[:, env].size()[0]):
                loss_pol+=self.log_prob_buffer[i, env]*(self.gae_buffer[i, env].detach())
        loss_pol*=-1
        self.pol_optimizer.zero_grad()
        loss_pol.backward()
        self.pol_optimizer.step()

    def value_update(self):
        loss_val = 0
        for env in range(args.num_envs):
            for i in range(self.reward_buffer[:, env].size()[0]):
                loss_val += (self.return_buffer[i, env]-self.value_buffer[i, env])**2
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
            # Point of contention
            val_buffer[i] = val_buffer[i].detach()
        val_buffer = np.array(val_buffer.detach())
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
        Y = self.update_steps_buffer
        Y = self.moving_average(Y, n=50)
        ax[0, 2].plot(X, Y)
        Y = np.cumsum(np.array(self.update_steps_buffer))
        ax[1, 0].plot(X, Y)
        Y = np.cumsum(np.array(self.update_time_buffer))
        ax[1, 1].plot(X, Y)        
        Y = self.log_avg_value
        ax[1, 2].plot(X, Y)        
        ax[0, 0].set_ylabel('Returns')
        ax[0, 0].set_ylim(bottom=0)
        ax[0, 1].set_ylabel('Rewards')
        ax[0, 2].set_ylabel("update Length")
        ax[1, 0].set_ylabel("# Steps")
        ax[1, 1].set_ylabel("Time (s)")
        ax[1, 2].set_ylabel("Average Value")
        plt.show()

    def ppo_update(self):
        loss_pol = 0
        size = min(len(self.reward_buffer), len(self.old_log_prob))
        print("size of prob_log buffer: ", len(self.log_prob_buffer), len(self.log_prob_buffer[0]))
        for i in range(size):
            logratio = self.log_prob_buffer[i] - self.old_log_prob[i].detach()
            ratio = torch.exp(logratio)
            tmp_loss1 = -1*self.gae_buffer[i].detach()*ratio
            tmp_loss2 = -1*self.gae_buffer[i].detach()*torch.clamp(ratio, 1-self.clip_coeff, 1+self.clip_coeff)
            loss_pol += torch.max(tmp_loss1, tmp_loss2).mean()

        loss_pol/=len(self.reward_buffer)
        self.pol_optimizer.zero_grad()
        loss_pol.backward()
        self.pol_optimizer.step()

    def train(self, seed=1):
        
        train_time = time.time()
        global_step=0
        next_done = torch.zeros(args.num_envs)
        num_upd = args.total_timesteps // args.batch_size
        obs = self.envs.reset()
        print(obs.shape)

        for update in range(1, num_upd+1):
            print("Update: ", update)
            # if args.anneal_lr:
            #     frac = 1.0 - (update - 1.0) / num_upd
            #     lrnow = frac * args.learning_rate
            #     self.pol_optimizer.param_groups[0]["lr"] = lrnow
            #     self.val_optimizer.param_groups[0]["lr"] = lrnow

            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32)
            # self.value_buffer.append(self.value.forward(obs_tensor))
            # done = False
            step_time = 0
            update_start = time.time()
            num_steps = 0
            for step in range(0, args.num_steps):
                global_step+=1*args.num_envs
                action = self.sample_action(obs, step=step)
                obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32)
                val = self.value.forward(obs_tensor)
                self.value_buffer[step, :] = torch.transpose(val, 0, 1)
                # self.value_buffer.append(val)
                obs, reward, done, info = self.envs.step(action)
                # self.reward_buffer.append(reward)
                self.reward_buffer[step, :] = torch.tensor(reward)
                step_dur = time.time()-update_start
                update_start = time.time()
                step_time+=step_dur
            
            self.update_steps_buffer.append(global_step)
            self.update_time_buffer.append(step_time)
            step_time/=global_step
            self.eps_run+=1
            update_time = time.time()
            update_dur = update_time - train_time
            train_time = time.time()

            ## Update 
            self.get_return_buffer()
            self.get_td_buffer()
            self.get_gae_buffer(lmbda=0.99)
            
            if update != 1:
                self.policy_update()
                self.value_update()
            self.log_data()
            # Cacheing previous policy's log buffer
            self.old_log_prob = self.log_prob_buffer
            # self.flush_post_ep()

            if (update) % 10 == 0:
                avg_reward = self.log_avg_reward[-1]
                print("update:", update, "Average Reward:", avg_reward, "Average Return: ", self.log_avg_return[-1])

            if (update) % 10 == 0 and self.plot:
                self.plot_training()
            
if __name__ == "__main__":
    args = parse_args()
    sim = Simulation(args)
    # sim.train()
    # reward = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 4, 5, 1, 2, 5, 3, 4, 8]])
    # reward = torch.ones((1600, 2))
    # values = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17, 18], [10, 11, 12, 13, 14, 15, 16, 17, 18]])
    # values = torch.ones((1600, 2))
    # values = torch.cumsum(values, dim=0)
    # values *= 2
    # reward = torch.transpose(reward, 0, 1)
    # values = torch.transpose(values, 0, 1)
    # reward = list(reward)
    # values = list(values)
    # print(reward)
    # sim.value_buffer=values
    # sim.reward_buffer=reward
    # sim.gamma=1
    # returns = sim.get_return_buffer()
    # sim.return_buffer = returns
    # sim.td_buffer = sim.get_td_buffer()
    # print("td buffer: ")
    # print(sim.td_buffer)
    # sim.gae_buffer = sim.get_gae_buffer(lmbda=)
    # print(sim.gae_buffer)

    # sim.gae_buffer = sim.get_gae_buffer(lmbda=1)
    # print(sim.gae_buffer)

    # print(returns[-20:, 0])
    # td_buffer = sim.get_td_buffer()
    # sim.td_buffer = td_buffer
    # gae_estimate = sim.get_gae_buffer(lmbda=0.99)
    sim.train()


