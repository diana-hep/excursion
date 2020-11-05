import numpy as np
import torch
from excursion.utils import mgrid, mesh2points
import math


def truth(x):
    return torch.square(x)


true_functions = [truth]

# Define threshold list
thresholds = torch.Tensor([0.0])

# Define grid for acquisition function
n_dims = 1


## rangedef[i] = [lower_i, upper_i, n_i] for i in n_dims
rangedef_1 = [-1, 1, 41]
rangedef = np.array([rangedef_1])

grid_1 = torch.linspace(
    start=rangedef_1[0], end=rangedef_1[1], steps=rangedef_1[2], dtype=torch.double
)
X = grid_1.view(-1, 1)

## Define grid for plotting, with same format as above, default same as X
plot_X = X

X_plot = np.linspace(rangedef_1[0], rangedef_1[1], rangedef_1[2])

mean_range = X  # default
plot_y = torch.Tensor([-5, 30])