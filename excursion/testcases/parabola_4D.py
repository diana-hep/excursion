import numpy as np
import torch
from excursion.utils import mgrid, mesh2points
import math


def truth(x):
	#paraboloid
    return torch.square(x).sum(dim=1)

true_functions = [truth]

# Define threshold list
thresholds = torch.Tensor([0.0])

# Define grid for acquisition function
n_dims = 4

## rangedef[i] = [lower_i, upper_i, n_i] for i in n_dims
rangedef = np.array([[-1, 1, 41], [-1, 1, 41], [-1, 1, 41],[-1, 1, 41]])

# meshgrid
plot_meshgrid = mgrid(rangedef)

# 2D points
X_plot = mesh2points(plot_meshgrid, rangedef[:, 2])
X = torch.from_numpy(X_plot)


def invalid_region(x):
    return np.array([False] * len(x))


