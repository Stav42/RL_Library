import random
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import argparse
import time
import torch.nn as nn
from torch.distributions.normal import Normal
import gymnasium as gym
from distutils.util import strtobool
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import wandb
from helper import parse_args
import datetime
from torch.utils.tensorboard import SummaryWriter

# set device to cpu or cuda


def make_env(gym_id, seed, rank, capture_video=None, run_name=None):
    def _init():
        env = gym.make(gym_id)
        if capture_video:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.reset(seed=seed+rank)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return _init

class Policy_Network(nn.Module):

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

class Value_Network(nn.Module):

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
    
    def __init__(self, args, render=False):

        self.env_id = "Ant-v4"
        # args.gym_id = "Ant-v4"
        self.num_cpu = 16
        self.args = args
        self.envs = SubprocVecEnv([make_env(self.args.gym_id, seed=i, rank=i) for i in range(self.args.num_envs)])
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.eps = 1e-6

        self.device = torch.device('cpu')
        if self.args.cuda == True and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            print("Device set to : cpu")
        
        self.obs_space_dim = self.envs.observation_space.shape[0]
        self.action_space_dim = self.envs.action_space.shape[0]

        self.policy = Policy_Network(self.obs_space_dim, self.action_space_dim).to(self.device)
        self.pol_optimizer = torch.optim.AdamW(self.policy.policy_net.parameters(), lr=self.learning_rate)

        self.value = Value_Network(self.obs_space_dim).to(self.device)
        self.val_optimizer = torch.optim.AdamW(self.value.value_net.parameters(), lr=self.learning_rate)

        self.log_prob_buffer = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        self.reward_buffer = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        self.return_buffer = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        self.steps = 0
        self.update_time_buffer = []
        self.update_steps_buffer = []
        self.value_buffer = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        self.td_buffer = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        self.gae_buffer = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        self.obs_buffer = torch.zeros((args.num_steps, args.num_envs, self.obs_space_dim)).to(self.device)
        self.action_buffer = torch.zeros((args.num_steps, args.num_envs, self.action_space_dim)).to(self.device)
        self.epsilon = 0
        self.wandb_run = None
        self.global_eps = 0
        self.episode_length = []

        self.log_avg_reward = []
        self.log_avg_return = []
        self.log_avg_value = []


        self.upd_rollout_time = 0
        self.upd_rollout_steps = 0

        self.old_log_prob = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.training_step = 0
        self.eps_run = 0

        self.clip_coeff = 0.2
        self.plot = True

        self.writer = None


    def tensorboard_init(self):
        run_name = f"{self.args.gym_id}__{self.args.description}__{self.args.exp_name}__{int(time.time())}"
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.writer.add_text(
            "Hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()]))
        )

    def get_flat_params_from(self, model: torch.nn.Module):
        return torch.cat([p.data.view(-1) for p in model.parameters()])
    
    #https://github.com/alirezakazemipour/TRPO-PyTorch/blob/main/common/utils.py
    def set_params(self, params: torch.nn.Module.parameters, model: torch.nn.Module):
        # print(f"Old Params are: {model.parameters()}")
        # print(f"New Params: {params}")
        pointer = 0
        for p in model.parameters():
            p.data.copy_(params[pointer:pointer+p.data.numel()].view_as(p.data))
            pointer += p.data.numel()

    def wandb_init(self):
        current_time_seconds = time.time()
        current_datetime = datetime.fromtimestamp(current_time_seconds)
        time_of_day = current_datetime.strftime("%H-%M")
        run_name = f"{self.args.gym_id}__{self.args.description}__{self.args.exp_name}__{time_of_day}"
        self.wandb_run = wandb.init(
            project=self.args.wandb_project_name, 
            entity=self.args.wandb_entity,
            sync_tensorboard=True,
            config=vars(self.args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    def action_gaussian(self, obs):
        obs = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device)
        mean, dev = self.policy.forward(obs)
        log_dev = torch.log(dev)
        return mean, dev, log_dev

    def sample_action(self, obs, action=None):
        obs = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device)
        mean, dev = self.policy.forward(obs)
        distrib = Normal(mean, dev)
        if action is None:
            action = distrib.sample()
        logp = distrib.log_prob(action).sum(axis=-1)
        return action, logp
    
    def save_model(self, path):
        path = path + "/Cartpole_PPO.pth"
        torch.save(self.policy.policy_net.state_dict(), path)

    def save_value(self, path):
        path = path + "/Cartpole_PPO_val.pth"
        torch.save(self.value.value_net.state_dict(), path)
    
    def flush_post_ep(self):
        self.log_prob_buffer    *= 0
        self.reward_buffer      *= 0
        self.return_buffer      *= 0
        self.value_buffer       *= 0
        self.td_buffer          *= 0
        self.gae_buffer         *= 0
        self.log_prob_buffer = self.log_prob_buffer.detach()
        self.reward_buffer = self.reward_buffer.detach()
        self.return_buffer = self.return_buffer.detach()
        self.value_buffer = self.value_buffer.detach()
        self.td_buffer = self.td_buffer.detach()
        self.gae_buffer = self.gae_buffer.detach()
        

    def flush_post_iter(self):
        self.flush_post_ep()
        self.log_avg_return.clear()
        self.log_avg_reward.clear()
        self.update_steps_buffer.clear()
        self.update_time_buffer.clear()
        self.log_avg_value.clear()

    def get_return_buffer(self, masks):
        gamma = self.gamma
        # for env in range(self.reward_buffer.size()[1]):
        for i, reward in enumerate(torch.flip(self.reward_buffer[:self.steps, :], [0])):
            if i == 0:
                self.return_buffer[self.steps-i-1, :] = reward
            else:
                self.return_buffer[self.steps-i-1, :] = reward + masks[self.steps-i-1, :]*self.return_buffer[self.steps-i, :]*gamma
        return self.return_buffer
    
    def get_gae_buffer(self, lmbda, masks):
        gamma = self.gamma
        l = self.steps
        for i in range(l):
            if i == 0:
                self.gae_buffer[l-i-1][:] = self.td_buffer[l-i-1][:]
            else:
                self.gae_buffer[l-i-1][:] = self.td_buffer[l-i-1][:].clone() + masks[l-i-1, :]*lmbda*gamma*self.gae_buffer[l-i][:].clone().detach()
        return self.gae_buffer

    def get_td_buffer(self, masks):
        gamma = self.gamma
        for i, rew in enumerate(self.reward_buffer[:self.steps, :]):
            if i == self.reward_buffer.size()[0]-1:
                self.td_buffer[i, :] = self.reward_buffer[i, :] - self.value_buffer[i, :].detach()
            else:
                self.td_buffer[i, :] = self.reward_buffer[i, :] + masks[i, :]*gamma*self.value_buffer[i+1, :].detach() - self.value_buffer[i, :].detach()
        return self.td_buffer
    
    def log_data(self):
        mean_rew = np.array(self.reward_buffer.cpu()).sum()
        self.log_avg_reward.append(mean_rew)
        mean_ret = np.array(self.return_buffer[:self.steps, :].cpu()).mean()
        self.log_avg_return.append(mean_ret)
        val_buffer = self.value_buffer
        for i, val in enumerate(range(len(val_buffer))):
            val_buffer[i] = val_buffer[i].detach()
        val_buffer = np.array(val_buffer.cpu().detach())
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

    def print_args_summary(self):
        print("Arguments passed to the script:")
        for arg, value in vars(self.args).items():
            print(f"{arg}: {value}")
        print(f"num_update: {self.args.total_timesteps//self.args.batch_size}")
        print(f"batch_size: {self.args.batch_size}")
    
    def test_functions(self):
        self.steps=10
        self.reward_buffer[:10, :] = 1
        print("Reward Buffer: ", self.reward_buffer.transpose(0, 1))
        for i in range(10):
            self.value_buffer[i, :] = i
            self.log_prob_buffer[i, :] = 2*i
        self.reward_buffer[:3, :] = 22
        time1 = time.time()
        self.get_return_buffer()
        return_time = time.time()-time1
        time2 = time.time()
        self.get_td_buffer()
        td_time = time.time()-time2
        time3 = time.time()
        self.get_gae_buffer(lmbda=0.99)
        gae_time = time.time()-time3
        loss_pol = -torch.sum(self.log_prob_buffer * self.gae_buffer)
        print("Return buffer: ", self.return_buffer.transpose(0, 1))
        print(f"Return Calculation Time: {return_time}")
        print("TD Buffer: ", self.td_buffer.transpose(0, 1))
        print(f"TD Buffer Calculation Time: {td_time}")
        print("GAE Buffer: ", self.gae_buffer.transpose(0, 1))
        print(f"GAE Buffer Calculation Time: {gae_time}")
        print("Loss calculated: ", loss_pol)