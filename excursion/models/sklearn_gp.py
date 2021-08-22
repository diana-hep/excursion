import sklearn
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from .base import ExcursionModel
import numpy as np


def get_kernel(ndim, kernel_type):
    if kernel_type == 'const_rbf':
        length_scale = [1.]*ndim
        kernel = ConstantKernel() * RBF(length_scale_bounds=[0.1, 100.0], length_scale=length_scale)
    elif kernel_type == 'tworbf_white':
        kernel = ConstantKernel() * RBF(length_scale_bounds=[1e-2, 100]) + \
                 ConstantKernel() * RBF(length_scale_bounds=[100., 1000.0]) + \
                 WhiteKernel(noise_level_bounds=[1e-7, 1e-4])
    elif kernel_type == 'onerbf_white':
        kernel = ConstantKernel() * RBF(length_scale_bounds=[1e-2, 100]) + WhiteKernel(noise_level_bounds=[1e-7, 1e-1])
    else:
        raise RuntimeError('unknown kernel')
    return kernel


class SKLearnGP(ExcursionModel):
    """This is a guassian process used to compute the excursion model. Most if not all excursion models will be a
    gaussian process.
    """
    def __init__(self, ndim, kernel_type='const_rbf', alpha=10**-7):
        self.kernel = get_kernel(ndim, kernel_type)
        self.gp_params = {
            'alpha': alpha,
            'n_restarts_optimizer': 10,
            'random_state': 1234
            }
        self.gp = GaussianProcessRegressor(kernel=self.kernel, **self.gp_params)
        self.epsilon = 0.0
        # alpha = 10 ** -7
        # self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10, alpha=10 ** -7, random_state=1234)
        # self.train_targets = np.empty((0, ndim))
        # self.train_inputs = np.empty((0,))

        # y_list = [np.empty((0,)) for f in probelm_details.functions]
        # self.train_targets = None

    def update_model(self, x, y):
        """
        Updates training data (does not re-fit model hyper-parameters).
        """

        inputs = x
        targets = y if x.shape[1] != 1 else y.flatten()

        if self.epsilon > 0.0:
            # Add noise if the likelihood had included epsilon>0 in algo options
            raise NotImplementedError("The current package only supports noiseless models for sklearn models")
        if not hasattr(self, "X_train") and not hasattr(self, "y_train"):  # No data; add it
            self.X_train = inputs
            self.y_train = targets.ravel()
            print("My first update")

        else:
            print("Updating the training data")
            self.X_train = np.concatenate((self.X_train, inputs), axis=0)

            y_train = np.concatenate((self.y_train, targets), axis=0)
            self.y_train = y_train.ravel()

        # # # Should actually check to see if the data is correct, TO BE IMPLEMENTED
        # self.set_train_data(inputs=inputs, targets=targets, strict=False)

    def fit_model(self, fit_optimizer):
        if not hasattr(self, "X_train") and not hasattr(self, "y_train"):  # No data; add it
            raise ValueError("The model's does not have any training data to fit with.")
        else:
            if hasattr(self.gp, "kernel_"):
                self.kernel.set_params(**(self.gp.kernel_.get_params()))
                self.gp = GaussianProcessRegressor(kernel=self.kernel, **self.gp_params).fit(self.X_train, self.y_train)
            else:
                self.gp.fit(self.X_train, self.y_train)


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


#
# class SKLearnGP(ExcursionModel, sklearn.gaussian_process.GaussianProcessRegressor):
#     """This is a guassian process used to compute the excursion model. Most if not all excursion models will be a
#     gaussian process.
#     """
#     def __init__(self, ndim, kernel_type='const_rbf', alpha=10**-7):
#         kernel = get_kernel(ndim, kernel_type)
#         super(SKLearnGP, self).__init__(kernel=kernel, n_restarts_optimizer=10, alpha=alpha, random_state=1234)
#         self.epsilon = 0.0
#         # alpha = 10 ** -7
#         # self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10, alpha=10 ** -7, random_state=1234)
#         # self.train_targets = np.empty((0, ndim))
#         # self.train_inputs = np.empty((0,))
#
#         # y_list = [np.empty((0,)) for f in probelm_details.functions]
#         # self.train_targets = None
#
#     def update_model(self, x, y):
#         """
#         Updates training data (does not re-fit model hyper-parameters).
#         """
#
#         inputs = x
#         targets = y if x.shape[1] != 1 else y.flatten()
#
#         if self.epsilon > 0.0:
#             # Add noise if the likelihood had included epsilon>0 in algo options
#             raise NotImplementedError("The current package only supports noiseless models for sklearn models")
#         if not hasattr(self, "X_train_") and not hasattr(self, "y_train_"):  # No data; add it
#             self.X_train_ = inputs
#             self.y_train_ = targets.ravel()
#             print("My first update")
#
#         else:
#             print("It was else my fist update")
#             self.X_train_ = np.concatenate((self.X_train_, inputs), axis=0)
#
#             y_train_ = np.concatenate((self.y_train_, targets), axis=0)
#             self.y_train_ = y_train_.ravel()
#
#         # # # Should actually check to see if the data is correct, TO BE IMPLEMENTED
#         # self.set_train_data(inputs=inputs, targets=targets, strict=False)
#
#     def fit_model(self, fit_optimizer):
#         if not hasattr(self, "X_train_") and not hasattr(self, "y_train_"):  # No data; add it
#             raise ValueError("The model's does not have any training data to fit with.")
#         else:
#             if hasattr(self, "kernel_"):
#                 self.kernel.set_params(**(self.kernel_.get_params()))
#                 self.fit(self.X_train_, self.y_train_)
#             else:
#                 self.fit(self.X_train_, self.y_train_)
