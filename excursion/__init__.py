import numpy as np
from scipy.linalg import cho_solve
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import Matern
import torch
import gpytorch 


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
                                  n_restarts_optimizer=100, #IRINA
                                  alpha=alpha,
                                  random_state=1234)

    gp.fit(X, y.ravel())
    print('mll', gp.log_marginal_likelihood_value_)
    return gp


def fit_hyperparams(gp, likelihood, optimizer: str='Adam'):
    training_iter = 100
    X_train = gp.train_inputs[0]
    y_train = gp.train_targets

    if(optimizer=='LBFGS'):
        optimizer = torch.optim.LBFGS([
            {'params': gp.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1, line_search_fn=None)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
        
        def closure():
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from gp
            output = gp(X_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f outputscale: %.3f  noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            gp.covar_module.base_kernel.lengthscale.item(),
            gp.covar_module. outputscale.item(),
            gp.likelihood.noise.item()
            ))
            return loss

        for i in range(training_iter):
            print('inside optimizer: i = ', i )
            optimizer.step(closure)


    if(optimizer=='Adam'):
        optimizer = torch.optim.Adam([
            {'params': gp.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1, eps=10e-6)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from gp
            output = gp(X_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.sum().backward()
            optimizer.step()
        



# def get_gp_gpytorch(X, y, alpha=10**-7, kernel_name='const_rbf'):
#     if kernel_name == 'const_rbf':
#         length_scale = [1.]*X.shape[-1]
#         kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior = length_scale))

#     else:
#         raise RuntimeError('unknown kernel')


#     likelihood = gpytorch.likelihoods.GaussianLikelihood()
#     gp = ExactGPModel(X, y.ravel(), likelihood)
    
#     #train
#     gp.train()
#     likelihood.train()
    
#     return ????

