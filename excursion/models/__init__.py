import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from .gp import ExcursionGP
from .fit import *


class ExcursionModel(object):
    def __init__(self, testcase, algorithmopts, ninit, device):
        self.init = False
        self.model, self.likelihood = init_gp(testcase, algorithmopts, ninit, device)


def init_gp( testcase, algorithmopts, ninit, device):
    likelihood_type = algorithmopts["likelihood"]["type"]
    modelopts = algorithmopts["model"]["type"]
    kernelopts = algorithmopts["model"]["kernel"]
    prioropts = algorithmopts["model"]["prior"]

    n_dims = testcase.n_dims
    epsilon = float(algorithmopts["likelihood"]["epsilon"])
    dtype = torch.float64

    #
    # TRAIN DATA
    #
    X_grid = torch.Tensor(testcase.X_plot).to(device, dtype)
    init_type = algorithmopts["init_type"]
    noise_dist = MultivariateNormal(torch.zeros(ninit), torch.eye(ninit))
    print(X_grid.size())
    if init_type == "random":
        indexs = np.random.choice(range(len(X_grid)), size=ninit, replace=False)
        X_init = X_grid[indexs].to(device, dtype)
        print(X_init)
        print(X_init.size())
        noises = epsilon * noise_dist.sample(torch.Size([])).to(device, dtype)
        y_init = testcase.true_functions[0](X_init).to(device, dtype) + noises
        print(y_init)
        print(y_init.size())
    #
    #
    # This is broken
    #
    #
    elif init_type == "worstcase":
        X_init = [X_grid[0]]
        X_init = torch.Tensor(X_init).to(device, dtype)
        noises = epsilon * noise_dist.sample(torch.Size([])).to(device, dtype)
        y_init = testcase.true_functions[0](X_init).to(device, dtype) + noises
    elif init_type == "custom":
        raise NotImplementedError("Not implemented yet")
    else:
        raise RuntimeError("No init data specification found")

    #
    # LIKELIHOOD
    #

    ## has bernoullilikelihood been used? if yes what was result if no then why not?

    if likelihood_type == "GaussianLikelihood":
        if epsilon > 0.0:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise=torch.tensor([epsilon])).to(device, dtype)
        elif epsilon == 0.0:
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=torch.tensor([epsilon])).to(device, dtype)
    else:
        raise RuntimeError("unknown likelihood")

    #
    # GAUSSIAN PROCESS
    #
    if modelopts == "ExactGP" and kernelopts == "RBF":
        model = ExcursionGP(X_init, y_init, likelihood, prioropts).to(device)

    elif modelopts == "GridGP" and kernelopts == "RBF":
        model = ExcursionGP(X_init, y_init, likelihood, prioropts, grid=testcase.rangedef).to(device)

    else:
        raise RuntimeError("unknown gpytorch model")

    # fit
    # print("X_init ", X_init)
    # print("y_init ", y_init)
    model.train()
    likelihood.train()
    fit_hyperparams(model, likelihood)

    return model, likelihood

