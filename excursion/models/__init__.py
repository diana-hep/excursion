import gpytorch
import torch
from . import priors


class ExactGP_RBF(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, prior):
        super(ExactGP_RBF, self).__init__(train_x, train_y, likelihood)
        # prior
        if prior == "Lineal":
            self.mean_module = priors.LinealMean(
                ndim=train_x.shape[1]
            )  # gpytorch.means.LinealMean()
        elif prior == "Constant":
            self.mean_module = gpytorch.means.ConstantMean()
        elif prior == "Circular":
            self.mean_module = priors.CircularMean(ndim=train_x.shape[1])
        else:
            raise NotImplementedError()

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_constraint=gpytorch.constraints.GreaterThan(lower_bound=0.1)
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GridGPRegression_RBF(gpytorch.models.ExactGP):
    def __init__(self, grid, train_x, train_y, likelihood, prior):
        super(GridGPRegression_RBF, self).__init__(train_x, train_y, likelihood)
        num_dims = train_x.size(-1)
        # prior
        if prior == "Lineal":
            self.mean_module = priors.LinealMean(ndim=train_x.shape[1])
        elif prior == "Constant":
            self.mean_module = gpytorch.means.ConstantMean()
        elif prior == "Circular":
            self.mean_module = priors.CircularMean(ndim=train_x.shape[1])
        else:
            raise NotImplementedError()
        self.covar_module = gpytorch.kernels.GridKernel(
            gpytorch.kernels.RBFKernel(), grid=grid
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
