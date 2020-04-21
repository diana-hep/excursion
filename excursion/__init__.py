import numpy as np
import torch
import gpytorch 
import excursion


def get_gp(X, y, **kwargs):

    likelihoodopts = kwargs['likelihood']['type']
    modelopts = kwargs['model']['type']
    kernelopts = kwargs['model']['kernel']

    #
    #LIKELIHOOD
    #
    epsilon = kwargs['likelihood']['epsilon']

    if(likelihoodopts == 'GaussianLikelihood'):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=torch.tensor([epsilon])) 
    #TODO: more types
    else:
        raise RuntimeError('unknown likelihood')

    #
    #GAUSSIAN PROCESS
    #

    if(modelopts == 'ExactGP' and kernelopts =='RBF'):
        model = ExactGPModel(X, y, likelihood)
    #TODO: more types
    else:
        raise RuntimeError('unknown gpytorch model')

    model.train()
    likelihood.train()
    excursion.fit_hyperparams(model, likelihood)
    return model


def init_traindata(testcase, init_type, n_initialize):
    #init training data: number and how to select init points
    X_grid = testcase.X_plot
    print('x_grid', X_grid.shape)

    if(init_type=='random'):
        indexs = np.random.choice(range(len(X_grid)), size = n_initialize, replace=False)
        X_train = [X_grid[i] for i in indexs]
        X_train = np.vstack(X_train)
        return torch.from_numpy(X_train)

    elif(init_type=='worstcase'):
        X_train = X_grid[0]
        return X_train

    elif(init_type=='custom'):
        raise NotImplementedError('Not implemented yet')

    else:
        RuntimeError('No init data specification found')



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
            #print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f outputscale: %.3f  noise: %.3f' % (
            #i + 1, training_iter, loss.item(),
            #gp.covar_module.base_kernel.lengthscale.item(),
            #gp.covar_module. outputscale.item(),
            #gp.likelihood.noise.item()
            #))
            return loss


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

            #print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f outputscale: %.3f  noise: %.3f' % (
            #i + 1, training_iter, loss.item(),
            #gp.covar_module.base_kernel.lengthscale.item(),
            #gp.covar_module. outputscale.item(),
            #gp.likelihood.noise.item()
            #))




#
# TYPES OF MODEL GAUSSIAN PROCESS
#

class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(\
                                      gpytorch.kernels.RBFKernel(\
                                      lengthscale_constraint=gpytorch.constraints.GreaterThan(lower_bound=0.1)
                                      ))                                               
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
