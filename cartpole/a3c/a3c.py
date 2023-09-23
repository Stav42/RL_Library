import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import torch.nn as nn
from torch.distributions.normal import Normal
import gymnasium as gym
from .a2c import Simulation
            
class ParallelSimulation(Simulation):
    
    def __init__(self, render=False):
        super().__init__(sim)


    

sim = ParallelSimulation()
print("Simulation instantiated")

for seed in range(8):
    sim.train(num_eps=3000, seed=seed)
    sim.plot_training()
    sim.save_model(path="/Users/stav.42/RL_Library/cartpole/weights/")
    # sim.flush_post_iter()
sim.save_model(path="/Users/stav.42/RL_Library/cartpole/weights/")