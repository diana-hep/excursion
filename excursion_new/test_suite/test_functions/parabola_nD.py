import torch

def truth(x):
    # paraboloid
    return torch.square(x).sum(dim=1)


true_functions = [truth]

# Define threshold list
thresholds = [1.0]



# Define grid for acquisition function
## rangedef[i] = [lower_i, upper_i, n_i] for i in n_dims
# bounding_box = [[-2, 2], [-2, 2], [-2, 2], [-2, 2]]
ndim = 4
bounding_box = [[-2, 2]]*ndim
plot_npoints = [20]*ndim