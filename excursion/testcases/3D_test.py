import numpy as np
import torch
import math
from excursion.utils import mgrid, mesh2points


def truth(x):
    xv, yv, zv = x[:, 0], x[:, 1], x[:, 2]
    """more wiggles in physics case"""

    return xv + yv + zv


true_functions = [truth]

# Define threshold list
thresholds = torch.Tensor([0.0])

# Define grid for acquisition function
n_dims = 3

## rangedef[i] = [lower_i, upper_i, n_i] for i in n_dims
rangedef = np.array([[-1, 1, 10], [-1, 1, 10], [0, 1, 10]])

# meshgrid
plot_meshgrid = mgrid(rangedef)

# 2D points
X_plot = mesh2points(plot_meshgrid, rangedef[:, 2])
X = torch.from_numpy(X_plot)


def invalid_region(x):
    return np.array([False] * len(x))


# acq_rd = np.array([[0.0, 1.5, 16], [0.0, 1.5, 16], [0.0, 1.5, 16]])
# acqG = utils.mgrid(acq_rd)
# acqX = utils.mesh2points(acqG, acq_rd[:, 2])

# mn_rd = np.array([[0.0, 1.5, 16], [0, 1.5, 16], [0.0, 1.5, 16]])
# mnG = utils.mgrid(mn_rd)
# meanX = utils.mesh2points(mnG, mn_rd[:, 2])
