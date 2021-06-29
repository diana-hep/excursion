import torch, gpytorch


def fit_hyperparams(gp, likelihood, optimizer: str = "Adam"):

    #### THIS HYPERPARAMETER MATTERS A LOT.

    training_iter = 50
    X_train = gp.train_inputs[0]
    y_train = gp.train_targets

    if optimizer == "Adam":

        optimizer = torch.optim.Adam(
            [{"params": gp.parameters()}, ],  # Includes GaussianLikelihood parameters
            lr=0.1,
            eps=10e-6,
        )

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)


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

    else:
        raise NotImplementedError
