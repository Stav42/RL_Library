from baselearner import Simulation
from helper import parse_args
import gymnasium as gym
import torch
import numpy as np
import time
import torch

class A2CSimulation(Simulation):

    def __init__(self, args):
        super().__init__(args)

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

    def train(self, seed=1):
        
        global_step=0
        self.global_eps=0
        num_upd = args.total_timesteps // args.batch_size
        obs = self.envs.reset()

        for update in range(1, num_upd+1):
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
                action, log_probability = self.sample_action(obs)
                self.log_prob_buffer[step, :] = log_probability
                obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32)
                val = self.value.forward(obs_tensor)
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
            self.get_return_buffer(masks)
            self.get_td_buffer(masks)
            self.get_gae_buffer(lmbda=0.99, masks=masks)
            self.policy_update_single()
            self.value_update_single()
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
    sim = A2CSimulation(args)
    print(sim.gamma)
    sim.print_args_summary()
    sim.train()
