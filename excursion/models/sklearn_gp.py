from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from .base import ExcursionModel

def get_kernel(ndim, alpha=10**-7, kernel_name='const_rbf'):
    if kernel_name == 'const_rbf':
        length_scale = [1.]*ndim
        kernel = ConstantKernel() * RBF(length_scale_bounds=[0.1, 100.0], length_scale = length_scale)
    elif kernel_name == 'tworbf_white':
        kernel = ConstantKernel() * RBF(length_scale_bounds=[1e-2,100]) + \
                 ConstantKernel() * RBF(length_scale_bounds=[100., 1000.0]) + \
                 WhiteKernel(noise_level_bounds=[1e-7,1e-4])
    elif kernel_name == 'onerbf_white':
        kernel = ConstantKernel() * RBF(length_scale_bounds=[1e-2,100]) + WhiteKernel(noise_level_bounds=[1e-7,1e-1])
    else:
        raise RuntimeError('unknown kernel')
    return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=alpha, random_state=1234)

class SKLearnGP(ExcursionModel):
    """This is a guassian process used to compute the excursion model. Most if not all excursion models will be a
    gaussian process.
    """
    def __init__(self, kernel_type):
        self.gp = get_gp()

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
        if self.epsilon > 0.0:
            # Add noise if the likelihood had included epsilon>0 in algo options
            targets = targets + self.epsilon * MultivariateNormal(torch.zeros(len(targets)), torch.eye(len(targets)))\
                .sample(torch.Size([])).to(device=self.device, dtype=self.dtype)
        if self.train_inputs is not None and self.train_targets is not None:
            inputs = torch.cat(
                (self.train_inputs[0], inputs), dim=0)
            targets = torch.cat(
                (self.train_targets, targets), dim=0)
        self.set_train_data(inputs=inputs, targets=targets, strict=False)
        return self

    def fit_model(self, fit_optimizer):
        return fit_hyperparams(self, fit_optimizer)

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
