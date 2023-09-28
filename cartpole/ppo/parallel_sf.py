import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import time
import sys
import torch.nn as nn
from torch.distributions.normal import Normal
import gymnasium as gym
from distutils.util import strtobool
from typing import Optional
from torch.utils.tensorboard import SummaryWriter

import functools

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

from datetime import date
today = date.today()
d = str(today.strftime("%b-%d-%Y"))

def parse_args():
    parser =  argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=d, help="name of this experiment")
    parser.add_argument('--gym-id', type=str, default="Cartpole-V1", help="id of this environment")
    parser.add_argument('--learning-rate', type=float, default=2.5e-4, help="set learning rate for algorithm")
    parser.add_argument('--seed', type=int, default=1, help="seed of this experiment")
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs="?", const="True", help="To enable WandB tracking")
    parser.add_argument('--wandb-project-name', type=str)

    args = parser.parse_args()
    return args

ENVS = ['Cartpole_v4', 'Cartpole_v4']

def make_mujoco_env(env_name, _cfg, _env_config, render_mode: Optional[str] = None, **kwargs):
    env = gym.make(env_name, render_mode=render_mode)
    return env

def add_mujoco_env_args(env, parser):
    pass

def register_components():
    for env in ENVS:
        register_env(env.name, make_mujoco_env)

def mujoco_override_defaults(env, parser):
    parser.set_defaults(
        batched_sampling=False,
        num_workers=8,
        num_envs_per_worker=8,
        worker_num_splits=2,
        train_for_env_steps=10000000,
        encoder_mlp_layers=[64, 64],
        env_frameskip=1,
        nonlinearity="tanh",
        batch_size=1024,
        kl_loss_coeff=0.1,
        use_rnn=False,
        adaptive_stddev=False,
        policy_initialization="torch_default",
        reward_scale=1,
        rollout=64,
        max_grad_norm=3.5,
        num_epochs=2,
        num_batches_per_epoch=4,
        ppo_clip_ratio=0.2,
        value_loss_coeff=1.3,
        exploration_loss_coeff=0.0,
        learning_rate=0.00295,
        lr_schedule="linear_decay",
        shuffle_minibatches=False,
        gamma=0.99,
        gae_lambda=0.95,
        with_vtrace=False,
        recurrence=1,
        normalize_input=True,
        normalize_returns=True,
        value_bootstrap=True,
        experiment_summaries_interval=3,
        save_every_sec=15,
        serial_mode=False,
        async_rl=False,
    )

def parse_mujoco_cfg(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_mujoco_env_args(partial_cfg.env, parser)
    mujoco_override_defaults(partial_cfg.env, parser)
    final_cfg=parse_full_cfg(parser, argv)
    return final_cfg



def main():
    register_components()
    cfg = parse_mujoco_cfg()
    status = run_rl(cfg)


if __name__ == "__main__":
    main()