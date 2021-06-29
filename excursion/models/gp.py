from gpytorch.models import ExactGP
from .kernel import Kernel
import numpy as np
from .fit import *


def get_gp(X, y, likelihood, algorithmopts, testcase, device):
    modelopts = algorithmopts["model"]["type"]
    kernelopts = algorithmopts["model"]["kernel"]
    prioropts = algorithmopts["model"]["prior"]

    #
    # GAUSSIAN PROCESS
    #

    # to
    X = X.to(device)
    y = y.to(device)

    #
    # GAUSSIAN PROCESS
    #
    if modelopts == "ExactGP" and kernelopts == "RBF":
        model = ExcursionGP(X, y, likelihood, prioropts).to(device)
    elif modelopts == "GridGP" and kernelopts == "RBF":
        model = ExcursionGP(X, y, likelihood, prioropts, grid=testcase.rangedef).to(device)
    else:
        raise RuntimeError("unknown gpytorch model")

    # fit
    model.train()
    likelihood.train()
    fit_hyperparams(model, likelihood)

    return model


class ExcursionGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, prior, grid = None):
        super(ExcursionGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if grid is None:
            self.covar_module = Kernel(model_type='ScaleKernel', base_kernel='RBFKernel').get_kernel()

        else:
            grid_bounds = grid[:, :-1]
            grid_n = grid[:, -1]
            grid = torch.zeros(int(np.max(grid_n)), len(grid_bounds), dtype=torch.double)

            for i in range(len(grid_bounds)):
                grid[:, i] = torch.linspace(
                    grid_bounds[i][0], grid_bounds[i][1], int(grid_n[i]), dtype=torch.double
                )

            self.covar_module = Kernel(model_type='GridKernel', base_kernel='RBFKernel', grid = grid).get_kernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
