import numpy as np
from scipy.linalg import cho_solve
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import Matern


def get_gp(X, y, alpha=10**-7, kernel_name='const_rbf'):
    if kernel_name == 'const_rbf':
        length_scale = [1.]*X.shape[-1]
        kernel = ConstantKernel() * RBF(length_scale_bounds=[0.1, 100.0], length_scale = length_scale)
    elif kernel_name == 'tworbf_white':
        kernel = ConstantKernel() * RBF(length_scale_bounds=[1e-2,100]) + \
                 ConstantKernel() * RBF(length_scale_bounds=[100., 1000.0]) + \
                 WhiteKernel(noise_level_bounds=[1e-7,1e-4])
    elif kernel_name == 'onerbf_white':
        kernel = ConstantKernel() * RBF(length_scale_bounds=[1e-2,100]) + WhiteKernel(noise_level_bounds=[1e-7,1e-1])
    else:
        raise RuntimeError('unknown kernel')
    gp = GaussianProcessRegressor(kernel=kernel,
                                  n_restarts_optimizer=10,
                                  alpha=alpha,
                                  random_state=1234)
    gp.fit(X, y.ravel())
    return gp

