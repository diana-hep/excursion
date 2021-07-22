from excursion_new.sampler import *
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
import gpytorch
from excursion_new.models import ExcursionGP, fit_hyperparams, ExcursionModel, ExactGP
from excursion_new.acquisition import MES, AcquisitionFunction


def build_sampler(generator: SampleGenerator, **kwargs):
    """Build a default random sample generator.
     For the special generator called "random" the return value is None.
     Parameters
     ----------
     generator : "random", "latin_sample", "latin_hypercube" \
             or SampleGenerator instance"
         Should inherit from `skopt.sampler.SampleGenerator`.
     kwargs : dict
         Extra parameters provided to the generator at init time.
     """
    if generator is None:
        generator = "random"
    elif isinstance(generator, str):
        generator = generator.lower()
        allowed_genators = ["random"]
        if generator not in allowed_genators:
            raise ValueError("Valid strings for the generator parameter "
                             " are: 'latin', 'latin_hypercube', or 'random' not "
                             "%s." % generator)
    elif not isinstance(generator, SampleGenerator):
        raise ValueError("generator has to be an SampleGenerator."
                         "Got %s" % (str(type(generator))))

    if isinstance(generator, str):
        if generator == "random":
            generator = RandomChoice()
    generator.set_params(**kwargs)

    return generator


def build_acquisition_func(acq_function: str or AcquisitionFunction, **kwargs):
    """Build an acquisition function.
     For the special acq_function called "random" the return value is None.
     Parameters
     ----------
     function : "MES", "PES", or AcquisitionFunction instance"
         Should inherit from `skopt.sampler.SampleGenerator`.
     kwargs : dict
         Extra parameters provided to the acq_function at init time.
     """
    if acq_function is None:
        acq_function = "MES"
    elif isinstance(acq_function, str):
        acq_function = acq_function.lower()
        allowed_genators = ["mes"]
        if acq_function not in allowed_genators:
            raise ValueError("Valid strings for the acq_function parameter "
                             " are: 'MES', or 'PES' not %s." % acq_function)
    elif not isinstance(acq_function, AcquisitionFunction):
        raise TypeError("acq_function has to be an AcquisitionFunction."
                         "Got %s" % (str(type(acq_function))))

    if isinstance(acq_function, str):
        if acq_function == "mes":
            acq_function = MES(device=torch.device('cuda'), dtype=torch.float64)
    acq_function.set_params(**kwargs)

    return acq_function


def build_model(model: str or ExcursionModel, init_X=None, init_y=None, **kwargs):
    """Build an acquisition function.
     For the special acq_function called "random" the return value is None.
     Parameters
     ----------
     model : "ExactGP", "GridGP", or ExcursionModel instance"
         Should inherit from `excursion.models.ExcursionModel`.
     kwargs : dict
         Extra parameters provided to the acq_function at init time.
     """
    if model is None:
        model = "exactgp"
    elif isinstance(model, str):
        model = model.lower()
        allowed_models = ["exactgp"]
        if model not in allowed_models:
            raise ValueError("Valid strings for the model parameter "
                             " are: 'ExactGP', or 'GridGP' not %s." % allowed_models)
    elif not isinstance(model, ExcursionModel):
        raise TypeError("acq_function has to be an ExcursionModel."
                         "Got %s" % (str(type(model))))

    if isinstance(model, str):
        if model == "exactgp":
            epsilon = 0.0
            noise_dist = MultivariateNormal(torch.zeros(1), torch.eye(1))
            noises = epsilon * noise_dist.sample(torch.Size([])).to(device=torch.device('cuda'), dtype=torch.float64)
            init_y = init_y.to(device=torch.device('cuda'), dtype=torch.float64) + noises
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.tensor([epsilon])).to(device=torch.device('cuda'), dtype=torch.float64)
            model = ExactGP(init_X, init_y, likelihood).to(device=torch.device('cuda'), dtype=torch.float64)
            likelihood = model.likelihood
            model.train()
            likelihood.train()
            fit_hyperparams(model)

    model.set_params(**kwargs)

    return model


def build_model_init(base_estimator: str, X_init, device, dtype, n_init_points, true_function):
    X_init = torch.from_numpy(X_init).to(device=device, dtype=dtype)
    epsilon = 0.0
    noise_dist = MultivariateNormal(torch.zeros(n_init_points), torch.eye(n_init_points))
    noises = epsilon * noise_dist.sample(torch.Size([])).to(device=device, dtype=dtype)
    y_init = true_function(X_init).to(device=device, dtype=dtype) + noises
    if epsilon > 0.0:
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise=torch.tensor([epsilon])).to(device=device, dtype=dtype)
    elif epsilon == 0.0:
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=torch.tensor([epsilon])).to(device, dtype)
    if base_estimator == "ExactGP":
        model = ExcursionGP(X_init, y_init, likelihood).to(device=device, dtype=dtype)
    model.train()
    likelihood = model.likelihood.train()
    likelihood.train()
    fit_hyperparams(model)

    return model


def build_model_old(base_estimator: str, X_init, y_init, device, dtype, n_init_points=1):
    # X_init = torch.from_numpy(X_init).to(device=device, dtype=dtype)
    epsilon = 0.0
    noise_dist = MultivariateNormal(torch.zeros(n_init_points), torch.eye(n_init_points))
    noises = epsilon * noise_dist.sample(torch.Size([])).to(device=device, dtype=dtype)
    y_init = y_init.to(device=device, dtype=dtype) + noises
    if epsilon > 0.0:
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise=torch.tensor([epsilon])).to(device=device, dtype=dtype)
    elif epsilon == 0.0:
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=torch.tensor([epsilon])).to(device, dtype)
    if base_estimator == "ExactGP":
        model = ExcursionGP(X_init, y_init, likelihood).to(device=device, dtype=dtype)
    model.train()
    likelihood = model.likelihood
    likelihood.train()
    fit_hyperparams(model)

    return model

