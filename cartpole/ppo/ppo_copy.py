from baselearner import Simulation
from helper import parse_args
import gymnasium as gym
import torch
import numpy as np
import time
import torch

class PPOSimulation(Simulation):

    def __init__(self, args):
        super().__init__(args)

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
    sim = PPOSimulation(args)
    print(sim.gamma)
    sim.print_args_summary()
    sim.train()
