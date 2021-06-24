import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import gpytorch
import excursion
from excursion.models import ExactGP_RBF, GridGPRegression_RBF
from excursion.acquisition import acquisition_functions


def init_gp(testcase, algorithmopts, ninit, device):
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

    if init_type == "random":
        indexs = np.random.choice(range(len(X_grid)), size=ninit, replace=False)
        X_init = X_grid[indexs].to(device, dtype)
        noises = epsilon * noise_dist.sample(torch.Size([])).to(device, dtype)
        y_init = testcase.true_functions[0](X_init).to(device, dtype) + noises
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
                noise=torch.tensor([epsilon])
            ).to(device, dtype)
        elif epsilon == 0.0:
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=torch.tensor([epsilon])
            ).to(device, dtype)

    else:
        raise RuntimeError("unknown likelihood")

    #
    # GAUSSIAN PROCESS
    #
    if modelopts == "ExactGP" and kernelopts == "RBF":
        model = ExactGP_RBF(X_init, y_init, likelihood, prioropts).to(device)
    elif modelopts == "GridGP" and kernelopts == "RBF":
        grid_bounds = testcase.rangedef[:, :-1]
        grid_n = testcase.rangedef[:, -1]

        grid = torch.zeros(int(np.max(grid_n)), len(grid_bounds), dtype=torch.double)

        for i in range(len(grid_bounds)):
            # a = torch.linspace(
            #    grid_bounds[i][0], grid_bounds[i][1], int(grid_n[i]), dtype=torch.double
            # )

            grid[:, i] = torch.linspace(
                grid_bounds[i][0], grid_bounds[i][1], int(grid_n[i]), dtype=torch.double
            )

        model = GridGPRegression_RBF(grid, X_init, y_init, likelihood, prioropts).to(
            device
        )

    else:
        raise RuntimeError("unknown gpytorch model")

    # fit
    print("X_init ", X_init)
    print("y_init ", y_init)
    model.train()
    likelihood.train()
    excursion.fit_hyperparams(model, likelihood)

    return model, likelihood


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

    if modelopts == "ExactGP" and kernelopts == "RBF":
        model = ExactGP_RBF(X, y, likelihood, prioropts).to(device)
    elif modelopts == "GridGP" and kernelopts == "RBF":
        grid_bounds = testcase.rangedef[:, :-1]
        grid_n = testcase.rangedef[:, -1]

        grid = torch.zeros(int(np.max(grid_n)), len(grid_bounds), dtype=torch.double)

        for i in range(len(grid_bounds)):
            # a = torch.linspace(
            #    grid_bounds[i][0], grid_bounds[i][1], int(grid_n[i]), dtype=torch.double
            # )

            grid[:, i] = torch.linspace(
                grid_bounds[i][0], grid_bounds[i][1], int(grid_n[i]), dtype=torch.double
            )

        model = GridGPRegression_RBF(grid, X, y, likelihood, prioropts).to(device)

    else:
        raise RuntimeError("unknown gpytorch model")

    # fit
    model.train()
    likelihood.train()
    fit_hyperparams(model, likelihood)

    return model


def fit_hyperparams(gp, likelihood, optimizer: str = "Adam"):
    training_iter = 100
    X_train = gp.train_inputs[0]
    y_train = gp.train_targets

    if optimizer == "LBFGS":
        optimizer = torch.optim.LBFGS(
            [{"params": gp.parameters()},],  # Includes GaussianLikelihood parameters
            lr=0.1,
            line_search_fn=None,
        )

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
            # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f outputscale: %.3f  noise: %.3f' % (
            # i + 1, training_iter, loss.item(),
            # gp.covar_module.base_kernel.lengthscale.item(),
            # gp.covar_module. outputscale.item(),
            # gp.likelihood.noise.item()
            # ))
            return loss

    if optimizer == "Adam":

        optimizer = torch.optim.Adam(
            [{"params": gp.parameters()},],  # Includes GaussianLikelihood parameters
            lr=0.1,
            eps=10e-6,
        )

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from gp
            output = gp(X_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.sum().backward(retain_graph=True)
            optimizer.step()

