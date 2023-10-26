from baselearner import Simulation
from helper import parse_args
from typing import Callable
import gymnasium as gym
import torch
from torch.autograd import Variable
import numpy as np
import time

class TRPOSimulation(Simulation):

    def __init__(self, args):
        super().__init__(args)

    def get_kl(self, inds):
        b_obs = self.obs_buffer.reshape((-1,) + self.envs.observation_space.shape)
        mean1, std1, log_std1 = self.action_gaussian(b_obs[inds])

        mean0 = Variable(data=mean1)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)
    

    def fisher_vector_product(self, inds, y):
        kl = self.get_kl(inds)
        kl = kl.mean()
        
        grads = torch.autograd.grad(kl, self.policy.policy_net.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([g.view(-1) for g in grads])
        # print("flat_grads: ", flat_grads)
        # print("y: ", y)
        kl_v = (flat_grad_kl * Variable(y)).sum()
        # inner_prod = flat_grads.t()@y
        # print(inner_prod)
        grads = torch.autograd.grad(kl_v, self.policy.policy_net.parameters(), retain_graph=True)
        flat_grad_grad_kl = torch.cat([g.reshape(-1) for g in grads]).data
        return flat_grad_grad_kl + y*args.damping
    
    def cg(self, A: Callable, b: torch.Tensor, steps: int, inds, tol: float=1e-6) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b - A(inds, x)
        d = r.clone()
        tol_new = r.t()@r
        for _ in range(steps):
            if tol_new<tol:
                break
            q = A(inds, d)
            alpha = tol_new/(d.t()@q)
            x+= alpha*d
            r-= alpha*q 
            tol_old = tol_new.clone()
            tol_new = r.t()@r
            beta = tol_new/tol_old
            d = r + beta*d
        return x

    #http://joschu.net/blog/kl-approx.html
    def learn(self):
        b_inds = np.arange(args.batch_size)
        b_obs = self.obs_buffer.reshape((-1,) + self.envs.observation_space.shape)
        b_actions = self.action_buffer.reshape((-1,) + self.envs.action_space.shape)
        b_logprob = self.log_prob_buffer.reshape(-1)
        b_gae = self.gae_buffer.reshape(-1)
        b_val = self.value_buffer.reshape(-1)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            _, new_logprob = self.sample_action(b_obs[b_inds], b_actions[b_inds])
            new_val = self.value.forward(b_obs[b_inds])
            logratio = new_logprob - b_logprob[b_inds]
            ratio = logratio.exp()
            b_advantages = b_gae[b_inds]
            pg_loss = -b_advantages.detach()*ratio
            pg_loss = pg_loss.mean()
            # kl = logratio.mean()
            value_loss = ((new_val - b_val[b_inds])**2).mean()

            grads = torch.autograd.grad(pg_loss, self.policy.policy_net.parameters(), retain_graph=True)
            flatten_grads = torch.cat([g.view(-1) for g in grads]).data
            stepdir = self.cg(self.fisher_vector_product, inds=b_inds, b=-flatten_grads, steps=args.cg_steps)
            # print(f"stepdir: {stepdir.t()}")
            shs = 0.5 * (stepdir * self.fisher_vector_product(b_inds, stepdir)).sum(0, keepdim=True)

            # print(f"Quadratic Term: {shs}")
            beta = torch.sqrt(2 * args.trust_region / (shs + 1e-6))
            # print(f"Beta: {beta}")
            opt_step = beta*stepdir

            with torch.no_grad():
                old_loss = pg_loss
                params = self.get_flat_params_from(self.policy.policy_net)
                exp_alpha = 1
                params_done = False

                for i in range(self.args.line_num_search):
                    # print(f"Line Search Iteration: {i}")
                    new_params = params + opt_step*exp_alpha
                    # print(f"params: {params} \n opt_step: {opt_step}")
                    _, tmp_logprob = self.sample_action(b_obs[b_inds], b_actions[b_inds])
                    # print("Without setting params, tmp_logprob: ", tmp_logprob)
                    self.set_params(params=new_params, model=self.policy.policy_net)
                    # print("Size of b_inds: ", len(b_inds))
                    _, tmp_logprob = self.sample_action(b_obs[b_inds], b_actions[b_inds])
                    # print("After setting params, tmp_logprob: ", tmp_logprob)
                    tmp_logratio = (tmp_logprob-b_logprob[b_inds])
                    tmp_ratio = tmp_logratio.exp()
                    tmp_advantages = b_gae[b_inds]
                    tmp_loss = (-tmp_advantages.detach()*tmp_ratio).mean()
                    tmp_kl = ((tmp_ratio - 1) - tmp_logratio).mean()
                    improvement = -tmp_loss + old_loss
                    if tmp_kl < 1.5*self.args.trust_region and improvement>=0 and torch.isfinite(tmp_loss):
                        params_done = True
                        # print("Parameters changed for minibatch: ", end/args.minibatch_size)
                        break
                    exp_alpha = 0.5*exp_alpha
                if not params_done:
                    self.set_params(params, self.policy.policy_net)
        
            self.val_optimizer.zero_grad()
            value_loss.backward()
            self.val_optimizer.step()

    #http://joschu.net/blog/kl-approx.html
    def learn_prev(self):
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
                b_inds = b_inds[start:end]
                
                _, new_logprob = self.sample_action(b_obs[b_inds], b_actions[b_inds])
                new_val = self.value.forward(b_obs[b_inds])
                logratio = b_logprob[b_inds]-new_logprob
                ratio = logratio.exp()
                b_advantages = b_gae[b_inds]
                pg_loss = -b_advantages.detach()*ratio
                pg_loss = pg_loss.mean()
                kl = ((ratio - 1) - logratio).mean()
                # kl = logratio.mean()
                # print("KL Divergence: ", kl)
                value_loss = ((new_val - b_val[b_inds])**2).mean()

                grads = torch.autograd.grad(pg_loss, self.policy.policy_net.parameters(), retain_graph=True)
                flatten_grads = torch.cat([g.view(-1) for g in grads]).data
                opt_dir = self.cg(self.fisher_vector_product, kl, -flatten_grads, args.cg_steps)
                # print(f"Opt_dir: {opt_dir.t()}")
                quad_term = (opt_dir * self.fisher_vector_product(kl=kl, y=opt_dir)).sum(0, keepdim=True)
                # print(f"Quadratic Term: {quad_term}")
                beta = torch.sqrt(2 * args.trust_region / (quad_term + 1e-6))
                # print(f"Beta: {beta}")
                opt_step = beta*opt_dir

                with torch.no_grad():
                    old_loss = pg_loss
                    params = self.get_flat_params_from(self.policy.policy_net)
                    exp_alpha = 1
                    params_done = False

                    for i in range(self.args.line_num_search):
                        print(f"Line Search Iteration: {i}")
                        new_params = params + opt_step*exp_alpha
                        # print(f"params: {params} \n opt_step: {opt_step}")
                        _, tmp_logprob = self.sample_action(b_obs[b_inds], b_actions[b_inds])
                        # print("Without setting params, tmp_logprob: ", tmp_logprob)
                        self.set_params(params=new_params, model=self.policy.policy_net)
                        # print("Size of b_inds: ", len(b_inds))
                        _, tmp_logprob = self.sample_action(b_obs[b_inds], b_actions[b_inds])
                        # print("After setting params, tmp_logprob: ", tmp_logprob)
                        tmp_logratio = (b_logprob[b_inds]-tmp_logprob)
                        tmp_ratio = tmp_logratio.exp()
                        tmp_advantages = b_gae[b_inds]
                        tmp_loss = (-tmp_advantages.detach()*tmp_ratio).mean()
                        tmp_kl = ((tmp_ratio - 1) - tmp_logratio).mean()
                        improvement = -tmp_loss + old_loss
                        if tmp_kl < 1.5*self.args.trust_region and improvement>=0 and torch.isfinite(tmp_loss):
                            params_done = True
                            print("Parameters changed for minibatch: ", end/args.minibatch_size)
                            break
                        exp_alpha = 0.5*exp_alpha
                    if not params_done:
                        self.set_params(params, self.policy.policy_net)
            
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
            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device)
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
                

            # Rollout Finish
            step_time/=global_step
            self.eps_run+=1
            with torch.no_grad():
                self.get_return_buffer(masks)
                self.get_td_buffer(masks)
                self.get_gae_buffer(lmbda=0.99, masks=masks)

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
    sim = TRPOSimulation(args)
    print(sim.gamma)
    sim.print_args_summary()
    sim.train()
