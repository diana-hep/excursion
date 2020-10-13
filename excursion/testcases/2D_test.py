import numpy as np
import torch
from excursion.utils import mgrid, mesh2points


def truth(x):
    xv, yv = x[:, 0], x[:, 1]

    return xv + yv


true_functions = [truth]

# Define threshold list
thresholds = torch.Tensor([0.0])

# Define grid for acquisition function
n_dims = 2

## rangedef[i] = [lower_i, upper_i, n_i] for i in n_dims
rangedef = np.array([[0.0, 1.5, 41], [0.0, 1.5, 41]])

# meshgrid
plot_meshgrid = mgrid(rangedef)

# 2D points
X_plot = mesh2points(plot_meshgrid, rangedef[:, 2])
X = torch.from_numpy(X_plot)


def invalid_region(x):
    return np.array([False] * len(x))
