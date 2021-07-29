from .kernel import Kernel
from .base import ExcursionModel
import torch
import gpytorch


class ExactGP(ExcursionModel, gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = Kernel(model_type='ScaleKernel', base_kernel='RBFKernel').get_kernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def update_model(self, x, y):
        if x.shape[1] == 1:
            inputs_i = torch.cat(
                (self.train_inputs[0], x), dim=0).flatten()
            targets_i = torch.cat(
                (self.train_targets.flatten(), y.flatten()), dim=0).flatten()
        else:
            inputs_i = torch.cat(
                (self.train_inputs[0], x), dim=0)
            targets_i = torch.cat(
                (self.train_targets, y), dim=0).flatten()

        self.set_train_data(inputs=inputs_i, targets=targets_i, strict=False)

        return self

    def fit_model(self, fit):
        return fit(self)



