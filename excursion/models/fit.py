import torch
import gpytorch


def fit_hyperparams(gp, optimizer: str = "Adam"):

    likelihood = gp.likelihood
    likelihood.train()
    gp.train()

    # # # # THIS HYPERPARAMETER MATTERS A LOT.

    training_iter = 150
    X_train = gp.train_inputs[0]
    y_train = gp.train_targets

    if optimizer == "Adam" or optimizer is None:

        optimizer = torch.optim.Adam(
            [{"params": gp.parameters()}, ],  # Includes GaussianLikelihood parameters
            lr=0.1,
            eps=10e-6,
        )

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)

        # See notes for idea on stopping criterion

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from gp
            output = gp(X_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.sum().backward(retain_graph=True)
            optimizer.step()

    elif optimizer == "LBFGS":
        optimizer = torch.optim.LBFGS(
            [
                {"params": gp.parameters()},
            ],  # Includes GaussianLikelihood parameters
            lr=0.1,
            line_search_fn=None,
        )

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

        for i in range(training_iter):
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
            optimizer.step(closure)

    else:
        raise NotImplementedError
    return gp
