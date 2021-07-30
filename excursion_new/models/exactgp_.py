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
        """
        Updates training data (does not re-fit model hyper-parameters).

        :param torch.Tensor x: The new training inputs to add.
        :param torch.Tensor y: The new training targets to add.
        :param bool strict: (default True) If `True`, the new inputs and
            targets must have the same shape, dtype, and device
            as the current inputs and targets. Otherwise, any shape/dtype/device are allowed.
        """
        strict = False
        # inputs = x
        # targets = y.flatten()
        # self.set_train_data(inputs=inputs, targets=targets, strict=strict)
        inputs = x
        targets = y if x.shape[1] != 1 else y.flatten()
        if self.train_inputs is not None and self.train_targets is not None:
            inputs = torch.cat(
                (self.train_inputs[0], inputs), dim=0)
            targets = torch.cat(
                (self.train_targets, targets), dim=0)
        self.set_train_data(inputs=inputs, targets=targets, strict=False)
        return self

    def fit_model(self, fit):
        return fit(self)

    # def special_strict_check(self, inputs=None, targets=None):
    #     if inputs is not None:
    #         if torch.is_tensor(inputs):
    #             inputs = (inputs,)
    #         inputs = tuple(input_.unsqueeze(-1) if input_.ndimension() == 1 else input_ for input_ in inputs)
    #         for input_, t_input in zip(inputs, self.train_inputs or (None,)):
    #             for attr in {"dtype", "device"}:
    #                 expected_attr = getattr(t_input, attr, None)
    #                 found_attr = getattr(input_, attr, None)
    #                 if expected_attr != found_attr:
    #                     msg = "Cannot modify {attr} of inputs (expected {e_attr}, found {f_attr})."
    #                     msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
    #                     raise RuntimeError(msg)
    #         torch.concat(())
    #     if targets is not None:
    #         for attr in {"dtype", "device"}:
    #             expected_attr = getattr(self.train_targets, attr, None)
    #             found_attr = getattr(targets, attr, None)
    #             if expected_attr != found_attr:
    #                 msg = "Cannot modify {attr} of targets (expected {e_attr}, found {f_attr})."
    #                 msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
    #                 raise RuntimeError(msg)
