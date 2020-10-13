import torch
import numpy as np
from excursion.utils import mgrid, mesh2points
import math

# Define true functions


def function_1(X: torch.Tensor) -> torch.Tensor:
    """ Returns a torch tensor where the ith-element is the i-th true function evaluated at x"""

    f1 = 10 - 10 * (torch.tanh(X * 2) + 0.15 * torch.sin(X * 15))
    return f1


def function_2(X: torch.Tensor) -> torch.Tensor:
    """ Returns a torch tensor where the ith-element is the i-th true function evaluated at x"""

    f2 = X + 2.5 * torch.sin(X * 3)
    return f2


def function_3(X: torch.Tensor) -> torch.Tensor:
    """ Returns a torch tensor where the ith-element is the i-th true function evaluated at x"""

    f3 = 2 * X - 1
    return f3


true_functions = [function_2]

# Define threshold list
thresholds = torch.Tensor([2.0])

# Define grid for acquisition function
n_dims = 1

## rangedef[i] = [lower_i, upper_i, n_i] for i in n_dims
rangedef = np.array([[0.0, 5.0, 500]])

# meshgrid
plot_meshgrid = mgrid(rangedef)

# 2D points
X_plot = mesh2points(plot_meshgrid, rangedef[:, 2])
X = torch.from_numpy(X_plot)


def invalid_region(x):
    return np.array([False] * len(x))
