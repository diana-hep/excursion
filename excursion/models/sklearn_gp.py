import sklearn
from .base import ExcursionModel
from .utils import get_kernel
import numpy as np


class SKLearnGP(ExcursionModel, sklearn.gaussian_process.GaussianProcessRegressor):
    """This is a guassian process used to compute the excursion model. Most if not all excursion models will be a
    gaussian process.
    """
    def __init__(self, ndim, kernel_type='const_rbf', alpha=10**-7):
        kernel = get_kernel(ndim, kernel_type)
        self.gp_params = {
            'alpha': alpha,
            'n_restarts_optimizer': 10,
            'random_state': 1234
            }
        super(SKLearnGP, self).__init__(kernel=kernel,  **self.gp_params)

        self.epsilon = 0.0

    def update_model(self, x, y):
        """
        Updates training data (does not re-fit model hyper-parameters).
        """

        inputs = x
        targets = y if x.shape[1] != 1 else y.flatten()

        if self.epsilon > 0.0:
            # Add noise if the likelihood had included epsilon>0 in algo options
            raise NotImplementedError("The current package only supports noiseless models for sklearn models")
        if not hasattr(self, "X_train_") and not hasattr(self, "y_train_"):  # No data; add it
            self.X_train_ = inputs
            self.y_train_ = targets.ravel()

        else:
            self.X_train_ = np.concatenate((self.X_train_, inputs), axis=0)

            y_train_ = np.concatenate((self.y_train_, targets), axis=0)
            self.y_train_ = y_train_.ravel()

        # # # Should actually check to see if the data is correct, TO BE IMPLEMENTED
        # self.set_train_data(inputs=inputs, targets=targets, strict=False)

    def fit_model(self, fit_optimizer):
        if not hasattr(self, "X_train_") and not hasattr(self, "y_train_"):  # No data; add it
            raise ValueError("The model's does not have any training data to fit with.")
        else:
            if hasattr(self, "kernel_"):
                self.kernel.set_params(**(self.kernel_.get_params()))
                self.fit(self.X_train_, self.y_train_)
            else:
                self.fit(self.X_train_, self.y_train_)
