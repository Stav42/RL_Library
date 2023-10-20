import random
import matplotlib.pyplot as plt
import psutil
import numpy as np
import os
import torch
import argparse
import time
import pickle
import torch.nn as nn
from torch.distributions.normal import Normal
import gymnasium as gym
from distutils.util import strtobool
from datetime import datetime
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from typing import Optional
import wandb
from torch.utils.tensorboard import SummaryWriter
from debugging import check_values_same
from helper import parse_args
import functools

torch.set_printoptions(threshold=10000)


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

        self.log_prob_buffer = torch.zeros((args.num_steps, args.num_envs))
        self.reward_buffer = torch.zeros((args.num_steps, args.num_envs))
        self.return_buffer = torch.zeros((args.num_steps, args.num_envs))
        self.steps = 0
        self.update_time_buffer = []
        self.update_steps_buffer = []
        self.value_buffer = torch.zeros((args.num_steps, args.num_envs))
        self.td_buffer = torch.zeros((args.num_steps, args.num_envs))
        self.gae_buffer = torch.zeros((args.num_steps, args.num_envs))
        self.obs_buffer = torch.zeros((args.num_steps, args.num_envs, self.obs_space_dim))
        self.action_buffer = torch.zeros((args.num_steps, args.num_envs, self.action_space_dim))
        self.epsilon = 0
        self.wandb_run = None
        self.global_eps = 0
        self.episode_length = []

        self.log_avg_reward = []
        self.log_avg_return = []
        self.log_avg_value = []

        self.upd_rollout_time = 0
        self.upd_rollout_steps = 0

        self.old_log_prob = torch.zeros((args.num_steps, args.num_envs))
        self.training_step = 0
        self.eps_run = 0

        self.clip_coeff = 0.2
        self.plot = True

        self.writer = None


    def tensorboard_init(self):
        run_name = f"{args.gym_id}__{args.description}__{args.exp_name}__{int(time.time())}"
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.writer.add_text(
            "Hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
        )

    
    def wandb_init(self):
        current_time_seconds = time.time()
        current_datetime = datetime.fromtimestamp(current_time_seconds)
        time_of_day = current_datetime.strftime("%H-%M")
        run_name = f"{args.gym_id}__{args.description}__{args.exp_name}__{time_of_day}"
        self.wandb_run = wandb.init(
            project=args.wandb_project_name, 
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )


    def sample_action(self, obs, action=None):
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
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

    def flip_list(self, lt):
        length = len(lt)
        tmp_list = [0]*length
        for i, val in enumerate(lt):
            tmp_list[length-i-1] = val
        return tmp_list

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
        for i, rew in enumerate(self.reward_buffer[:self.steps, 0]):
            if i == self.reward_buffer.size()[0]-1:
                self.td_buffer[i, :] = rew - self.value_buffer[i, :].detach()
            else:
                self.td_buffer[i, :] = rew + masks[i, :]*gamma*self.value_buffer[i+1, :].detach() - self.value_buffer[i, :].detach()
        return self.td_buffer

    def policy_update_single(self):
        loss_pol = -torch.sum(self.log_prob_buffer * self.gae_buffer)
        self.pol_optimizer.zero_grad()
        loss_pol.backward()
        self.pol_optimizer.step()

    def value_update_single(self):
        mini_return = self.return_buffer
        mini_value = self.value_buffer
        loss_val =  (mini_return - mini_value)**2
        loss = torch.sum(loss_val)
        self.val_optimizer.zero_grad()
        loss.backward()
        self.val_optimizer.step()

    def log_data(self):
        mean_rew = np.array(self.reward_buffer).sum()
        self.log_avg_reward.append(mean_rew)
        mean_ret = np.array(self.return_buffer[:self.steps, :]).mean()
        self.log_avg_return.append(mean_ret)
        val_buffer = self.value_buffer
        for i, val in enumerate(range(len(val_buffer))):
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

    def print_args_summary(self):
        print("Arguments passed to the script:")
        for arg, value in vars(args).items():
            print(f"{arg}: {value}")
        print(f"num_update: {args.total_timesteps//args.batch_size}")
        print(f"batch_size: {args.batch_size}")
        

    def wandb_logging(self, title, table, x_title, y_title):
        table_rollout = wandb.Table(data=table, columns=[x_title, y_title])
        wandb.log(
            {title: wandb.plot.line(table_rollout, x_title, y_title, title=title)}
        )        

    def print_stats(self):
        print(f"Average Value: {self.value_buffer.mean()}")
        print(f"Return Value: {self.return_buffer.mean()}")
        print(f"Reward Value: {self.reward_buffer.mean()}")

    def learn(self):
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            b_obs = self.obs_buffer.reshape((-1,) + self.envs.observation_space.shape)
            b_actions = self.action_buffer.reshape((-1,) + self.envs.action_space.shape)
            b_logprob = self.log_prob_buffer.reshape(-1)
            b_gae = self.gae_buffer.reshape(-1)
            b_val = self.value_buffer.reshape(-1)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                _, new_logprob = self.sample_action(b_obs[mb_inds], b_actions[mb_inds])
                new_val = self.value.forward(b_obs[mb_inds])
                logratio = new_logprob - b_logprob[mb_inds]
                ratio = logratio.exp()
                mb_advantages = b_gae[mb_inds]
                pg_loss1 = -mb_advantages.detach()*ratio
                pg_loss2 = -mb_advantages.detach()*torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = ((new_val - b_val[mb_inds])**2).mean()
            
                self.pol_optimizer.zero_grad()
                pg_loss.backward()
                self.pol_optimizer.step()

                self.val_optimizer.zero_grad()
                value_loss.backward()
                self.val_optimizer.step()

                

    def train(self, seed=1):
        
        global_step=0
        self.global_eps=0

        num_upd = args.total_timesteps // args.batch_size
        obs = self.envs.reset()

        for update in range(1, num_upd+1):
            print(f"Update: {update}")
            obs = self.envs.reset()
            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32)
            step_time = 0
            update_start = time.time()
            done = False
            self.steps = 0
            env_steps = [0]*args.num_envs
            self.global_eps+=1
            masks = torch.zeros(args.num_steps, args.num_envs)
            for step in range(0, args.num_steps):
                global_step+=1*args.num_envs
                self.steps+=1
                with torch.no_grad():
                    action, log_probability = self.sample_action(obs)
                self.log_prob_buffer[step, :] = log_probability
                obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32)
                val = self.value.forward(obs_tensor)
                with torch.no_grad():
                    self.value_buffer[step, :] = torch.transpose(val, 0, 1)
                obs, reward, done, info = self.envs.step(action)
                self.obs_buffer[step, :] = obs_tensor
                self.action_buffer[step, :] = action
                for i in range(len(env_steps)):
                    env_steps[i] += 1
                self.reward_buffer[step, :] = torch.tensor(reward)
                step_dur = time.time()-update_start
                step_time+=step_dur
                masks[step] = torch.tensor([not term for term in done])
                if done.any() == True:
                    for index, status in enumerate(done):
                        if status == True:
                            self.episode_length.append(env_steps[index])
                            env_steps[index] = 0
                
            step_time/=global_step
            self.eps_run+=1
            with torch.no_grad():
                self.get_return_buffer(masks)
                self.get_td_buffer(masks)
                self.get_gae_buffer(lmbda=0.99, masks=masks)
            self.learn()
            self.log_data()
            self.flush_post_ep()

            if (update) % 10 == 0:
                avg_reward = self.log_avg_reward[-1]
                print("update:", update, "Average Reward:", avg_reward, "Average Return: ", self.log_avg_return[-1], "Episode Length: ", np.array(self.episode_length[-10:]).mean())

            if (update) % 10 == 0 and self.plot:
                # self.plot_training()
                continue
            
if __name__ == "__main__":
    args = parse_args()
    sim = Simulation(args)
    if args.track:
        sim.wandb_init()
        print("WandB initialization done")
        sim.tensorboard_init()
    sim.print_args_summary()
    print("Start training")
    print("WandB Project Name: ", args.wandb_project_name)
    sim.train()
    sim.save_model(path="./weights")
    print("Weights Saved!")
    sim.wandb_run.finish()

