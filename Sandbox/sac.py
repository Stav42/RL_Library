from Qlearner import Simulation
from helper import parse_args
from typing import Callable
import gymnasium as gym
import torch
from torch.autograd import Variable
import numpy as np
import time

class SACSimulation(Simulation):

    def __init__(self, args):
        super().__init__(args)

    #http://joschu.net/blog/kl-approx.html
    def learn(self):
        b_inds = np.arange(args.batch_size)
        b_obs = self.obs_buffer.reshape((-1,) + self.envs.observation_space.shape)
        b_actions = self.action_buffer.reshape((-1,) + self.envs.action_space.shape)
        b_logprob = self.log_prob_buffer.reshape(-1)
        b_gae = self.gae_buffer.reshape(-1)
        b_val = self.value_buffer.reshape(-1)
        b_target = self.next_target.reshape(-1)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            i = 0
            for start in range(0, args.batch_size, args.minibatch_size):
                i+=1
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # print(f"b_actions[mb_inds][0]: {b_actions[mb_inds[0]]}")
                # print(f"b_observation[mb_inds][0]: {b_obs[mb_inds[0]]}")

                concat = torch.cat((b_actions[mb_inds], b_obs[mb_inds]), dim=1)
                # print(f"Concatenated list: ", concat[0])
                Q1_values = self.Q1(concat).reshape(-1)
                Q2_values = self.Q2(concat).reshape(-1)
                # print(f"Q1_values: {Q1_values.shape}")
                # print(f"TargetQ: {b_target[mb_inds].shape}")
                Q1_loss = torch.nn.functional.mse_loss(Q1_values, b_target[mb_inds])
                Q2_loss = torch.nn.functional.mse_loss(Q2_values, b_target[mb_inds])

                Q_loss = Q1_loss + Q2_loss
                # print("Q_loss: ", Q_loss)

                self.q_optimizer.zero_grad()
                Q_loss.backward(retain_graph=True)
                self.q_optimizer.step()

                if i%args.pol_freq == 0:
                    for _ in range(args.pol_freq):
                        # print("Policy Update")
                        action, log_prob = self.sample_action(b_obs[mb_inds])
                        concat = torch.cat((action, b_obs[mb_inds]), dim=1)
                        Q1_pi = self.Q1(concat)
                        Q2_pi = self.Q2(concat)
                        min_Q_pi = torch.min(Q1_pi, Q2_pi)
                        actor_loss = ((args.alpha*log_prob)-min_Q_pi).mean()
                        # print("Actor Loss: ", actor_loss)
                        self.pol_optimizer.zero_grad()
                        actor_loss.backward(retain_graph=True)
                        self.pol_optimizer.step()

                if i%args.target_update_freq == 0:
                    # print("Target Update")
                    for param, target_param in zip(self.Q1.value_net.parameters(), self.Q1_target.value_net.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(self.Q2.value_net.parameters(), self.Q2_target.value_net.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)            


    def train(self, seed=1):
        
        global_step=0
        self.global_eps=0

        num_upd = args.total_timesteps // args.batch_size

        for update in range(1, num_upd+1):
            print(f"Update: {update}")
            obs = self.envs.reset()
            step_time = 0
            update_start = time.time()
            done = False
            self.steps = 0
            env_steps = [0]*args.num_envs
            self.global_eps+=1
            masks = torch.zeros(args.num_steps, args.num_envs)
            flag = 0
            for step in range(0, args.num_steps):
                obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device)
                self.obs_buffer[step, :] = obs_tensor
                global_step+=1*args.num_envs
                self.steps+=1
                with torch.no_grad():
                    action, log_probability = self.sample_action(obs)
                self.log_prob_buffer[step, :] = log_probability
                next_obs, reward, done, info = self.envs.step(action)
                env_index = list(np.arange(args.num_envs))
                if done.any() == True:
                    for index, status in enumerate(done):
                        if status == True:
                            self.next_obs_buffer[step, index] = torch.tensor(info[index]['terminal_observation']).to(self.device)
                            self.episode_length.append(env_steps[index])
                            env_steps[index] = 0
                            env_index.remove(index)
                self.next_obs_buffer[step, env_index] = torch.tensor(np.array(next_obs[env_index]), dtype=torch.float32).to(self.device)
                obs = next_obs
                self.action_buffer[step, :] = action
                for i in range(len(env_steps)):
                    env_steps[i] += 1
                self.reward_buffer[step, :] = torch.tensor(reward)
                step_dur = time.time()-update_start
                step_time+=step_dur
                masks[step] = torch.tensor([not term for term in done])

            # Rollout Finish
            step_time/=global_step
            self.eps_run+=1
            with torch.no_grad():
                self.get_return_buffer(masks)
                self.get_qtarget_buffer(masks)
            
            # Buffers Done
            # Learning
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
    sim = SACSimulation(args)
    print(sim.gamma)
    sim.print_args_summary()
    sim.train()
