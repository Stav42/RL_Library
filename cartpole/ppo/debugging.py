import torch    

def check_values_same(self, new, old):
    delta = torch.abs(new - old)
    scalar = 0.03
    num_torch = torch.lt(delta, 0.0003)
    print(f"No. of elements less than {scalar}", torch.sum(num_torch).item())