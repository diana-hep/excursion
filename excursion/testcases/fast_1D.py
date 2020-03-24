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
rangedef_1 = [-1,1,100]
rangedef = torch.Tensor([rangedef_1])

grid_1 = torch.linspace(start=rangedef_1[0], end=rangedef_1[1], steps=rangedef_1[2])
X = torch.Tensor(grid_1).view(-1,1)

## Define grid for plotting, with same format as above, default same as X
plot_X = X

mean_range = X #default
plot_y = torch.Tensor([-5,30])


# import numpy as np

# def truthfunc(X):
#     f = 15
#     return 10-10*(np.tanh(X*2) + 0.15*np.sin(X*f))

# truth_functions = [truthfunc]
# thresholds = [0.7]

# plot_rangedef = np.asarray([
#     [-1,1,100]
# ])
# plotX = np.linspace(*plot_rangedef[0]).reshape(-1,1)

# acq_rangedef = np.asarray([
#     [-1,1,100]
# ])
# acqX = np.linspace(*acq_rangedef[0]).reshape(-1,1)



# mean_rangedef = np.asarray([
#     [-1,1,100]
# ])
# meanX = np.linspace(*mean_rangedef[0]).reshape(-1,1)
# ndims = 1
# y_lim = [-5,30]
    
    