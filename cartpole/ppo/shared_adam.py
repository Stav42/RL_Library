
import torch

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=2.5e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state=self.state[p]
                