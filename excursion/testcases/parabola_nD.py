import numpy as np
import torch
from excursion.utils import mgrid, mesh2points
import sys
import yaml

def truth(x):
    # paraboloid
    return torch.square(x).sum(dim=1)


true_functions = [truth]

# Define threshold list
thresholds = torch.Tensor([1.0])

#n dimensions
n_dims = 4
#file = yaml.safe_load(open(sys.argv[4], "r"))
#n_dims = int(file['n'])

# Define grid for acquisition function
## rangedef[i] = [lower_i, upper_i, n_i] for i in n_dims
rangedef = np.array([[-2, 2, 10],] * n_dims)

# meshgrid
plot_meshgrid = mgrid(rangedef)

# 2D points
X_plot = mesh2points(plot_meshgrid, rangedef[:, 2])
X = torch.from_numpy(X_plot)


def invalid_region(x):
    return np.array([False] * len(x))
