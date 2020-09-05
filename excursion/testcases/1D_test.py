import torch
import numpy as np

# Define true functions


def function_1(X: torch.Tensor) -> torch.Tensor:
    """ Returns a torch tensor where the ith-element is the i-th true function evaluated at x"""

    f1 = X 
    return f1


true_functions = [function_1]

# Define threshold list
thresholds = torch.Tensor([0.0])

# gaussian noise to the black box evaluation
# if you want other than gaussian noise, modify the likelihood in the notebook
epsilon = 0.0


# acquisition function type
acq_type = "MES"

# Define grid for acquisition function
n_dims = 1

## rangedef[i] = [lower_i, upper_i, n_i] for i in n_dims
rangedef_1 = [-1, 1, 100]
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
