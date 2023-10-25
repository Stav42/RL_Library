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
from typing import Callable

def fisher_vector_product(self, kl, y):
    grads = torch.autograd.grad(kl, self.policy.policy_net.parameters(), create_graph=True, retain_graph=True)
    flat_grads = torch.cat([g.view(-1) for g in grads])
    # print(flat_grads)
    inner_prod = flat_grads.t()@y
    # print(inner_prod)
    grads = torch.autograd.grad(inner_prod, self.policy.policy_net.parameters(), retain_graph=True)
    flat_grads = torch.cat([g.reshape(-1) for g in grads]).data
    return flat_grads

def matrix_prod(b):
    A = torch.Tensor([[3, 0, 1], [0, 4, 2], [1, 2, 3]])
    print("b: ", b.t())
    print("A*b: ", A@b.t())
    return A@b

def cg(A: Callable, b: torch.Tensor, steps: int, tol: float=1e-6) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b - A(x)
    d = r.clone()
    tol_new = r.t()@r
    print(f"r: {r} \n tol_new: {tol_new}")
    for _ in range(steps):
        if tol_new<tol:
            break
        q = A(d)
        alpha = tol_new/(d.t()@q)
        x+= alpha*d
        r-= alpha*q 
        tol_old = tol_new.clone()
        tol_new = r.t()@r
        beta = tol_new/tol_old
        d = r + beta*d
    return x

A = torch.Tensor([[3, 0, 1], [0, 4, 2], [1, 2, 3]])
b = torch.Tensor([3, 0, 1])

print(f"x: {cg(A=matrix_prod, b=b, steps=10)}")