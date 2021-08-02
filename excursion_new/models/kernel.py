from gpytorch.kernels import *
from gpytorch.constraints import GreaterThan
import torch
import numpy as np
# Importlib instead of globals


class Kernel(object):
    # Extend options to **kwargs
    def __init__(self, model_type: str, base_kernel: str, grid=None, **kwargs):
        if grid is None:
            self.kernel = globals()[model_type](globals()[base_kernel](
                lengthscale_constraint=GreaterThan(lower_bound=0.1)))
        else:
            grid_bounds = grid[:, :-1]
            grid_n = grid[:, -1]
            grid = torch.zeros(int(np.max(grid_n)), len(grid_bounds))
            for i in range(len(grid_bounds)):
                grid[:, i] = torch.linspace(
                    grid_bounds[i][0], grid_bounds[i][1], int(grid_n[i]))

            self.kernel = globals()[model_type](globals()[base_kernel](), grid=grid)

    def get_kernel(self):
        return self.kernel
