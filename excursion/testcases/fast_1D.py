import torch

# Define true functions

def function_1(X: torch.Tensor) -> torch.Tensor :
	""" Returns a torch tensor where the ith-element is the i-th true function evaluated at x"""

	f1 = torch.Tensor(10-10*(torch.tanh(X*2) + 0.15*torch.sin(X*15)))
	return f1

true_functions = [function_1]

# Define threshold list 
thresholds = torch.Tensor([0.7])


# Define grid for acquisition function
n_dims = 1

## rangedef[i] = [lower_i, upper_i, n_i] for i in n_dims
rangedef_1 = [-1,1,101]
rangedef = torch.Tensor([rangedef_1])

grid_1 = torch.linspace(start=rangedef_1[0], end=rangedef_1[1], steps=rangedef_1[2])
X = torch.Tensor(grid_1).view(-1,1)

## Define grid for plotting, with same format as above, default same as X
plot_X = X

mean_range = X #default
plot_y = torch.Tensor([-5,30])
