import argparse
import os
from distutils.util import strtobool

from datetime import date
today = date.today()
d = str(today.strftime("%d-%b-%Y"))

def parse_args():
    parser =  argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=d, help="name of this experiment")
    parser.add_argument('--gym-id', type=str, default="InvertedPendulum-v4", help="id of this environment")
    parser.add_argument('--learning-rate', type=float, default=2.5e-4, help="set learning rate for algorithm")
    parser.add_argument('--seed', type=int, default=1, help="seed of this experiment")
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs="?", const="True", help="To enable WandB tracking")
    parser.add_argument('--wandb_project_name', type=str, default="Prototype", help="the wandb's project name")
    parser.add_argument('--wandb_entity', type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument('--num_minibatches', type=int, default=32, help="the number of mini-batches")
    parser.add_argument('--num_envs', type=int, default=8, help="number of environments to activate")
    parser.add_argument('--total_timesteps', type=int, default=20000000, help="number of global steps to execute")
    parser.add_argument('--num_steps', type=int, default=800, help="the number of steps to run in each environment per policy rollout")
    parser.add_argument('--update_epochs', type=int, default=7, help="the K epochs to update the policy")
    parser.add_argument('--description', type=str, required=True, help="One-word-description for experiment name")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--cuda", type=bool, default=False, help="Toggle CUDA if available")
    parser.add_argument("--cg-steps", type=int, default=10, help="Number of iterations for Conjugate Gradient")
    parser.add_argument("--trust-region", type=float, default=0.001, help="Trust Region delta")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

# num_envs: Number of environments to activate
# num_steps: Number of steps rollout in each environment
# num_minibatches: Number of mini-batches
# args.batch_size: int(num_envs * num_steps)
# args.minibatch_size: args.batch_size//num_minibatches