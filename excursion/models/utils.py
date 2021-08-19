from gpytorch.kernels import *
from gpytorch.constraints import GreaterThan
import torch
import numpy as np
# Importlib instead of globals

#
# class Kernel(object):
#     # Extend options to **kwargs
#     def __init__(self, model_type: str, base_kernel: str, input_rangedef=None, **kwargs):
#         if input_rangedef is None:
#             self.kernel = globals()[model_type](globals()[base_kernel](
#                 lengthscale_constraint=GreaterThan(lower_bound=0.1)))
#         else:
#             grid_bounds = input_rangedef[:, :-1]
#             grid_n = input_rangedef[:, -1]
#             input_rangedef = torch.zeros(int(np.max(grid_n)), len(grid_bounds))
#             for i in range(len(grid_bounds)):
#                 input_rangedef[:, i] = torch.linspace(
#                     grid_bounds[i][0], grid_bounds[i][1], int(grid_n[i]))
#
#             self.kernel = globals()[model_type](globals()[base_kernel](), grid=input_rangedef)
#
#     def get_kernel(self):
#         return self.kernel

def get_kernel(model_type: str, base_kernel: str, input_rangedef=None):
    if input_rangedef is None:
        kernel = globals()[model_type](globals()[base_kernel](
            lengthscale_constraint=GreaterThan(lower_bound=0.1)))
    else:
        grid_bounds = input_rangedef[:, :-1]
        grid_n = input_rangedef[:, -1]
        input_rangedef = torch.zeros(int(np.max(grid_n)), len(grid_bounds))
        for i in range(len(grid_bounds)):
            input_rangedef[:, i] = torch.linspace(
                grid_bounds[i][0], grid_bounds[i][1], int(grid_n[i]))

        kernel = globals()[model_type](globals()[base_kernel](), grid=input_rangedef)
    return kernel