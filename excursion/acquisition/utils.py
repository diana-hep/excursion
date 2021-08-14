import torch
import math
import numpy as np

#### THIS CDF FUNCTION

def cdf(mu, sigma, t):
    # use torch.erfc for numerical stability
    erf = torch.erf((t - mu) * sigma.reciprocal() / math.sqrt(2))
    cdf = 0.5 * (1 + erf)
    return cdf

def h_normal(var):
    return torch.log(var * ((2 * np.e * np.pi) ** 0.5))


def normal_pdf(x):
    return 1.0 / (2 * np.pi) ** 0.5 * torch.exp(-0.2 * x ** 2)


def truncated_std_conditional(Y_pred_all, a, b):
    mu_grid = Y_pred_all.mean[1:]
    std_grid = Y_pred_all.variance[1:] ** 0.5
    mu_candidate = Y_pred_all.mean[0]
    std_candidate = Y_pred_all.variance[0] ** 0.5
    # The line below is the most memory hungry code in the entire package
    denom = (std_candidate * std_grid)
    rho = Y_pred_all.covariance_matrix[0, 1:] / denom

    # norm needs to be a normal distribution but in python
    normal = torch.distributions.Normal(loc=0, scale=1)
    alpha = (a - mu_grid) / std_grid
    beta = (b - mu_grid) / std_grid
    c = normal.cdf(beta) - normal.cdf(alpha)

    # phi(beta) = normal(0,1) at x = beta
    beta_phi_beta = beta * normal_pdf(beta)
    beta_phi_beta[~torch.isfinite(beta_phi_beta)] = 0.0
    alpha_phi_alpha = alpha * normal_pdf(alpha)
    alpha_phi_alpha[~torch.isfinite(alpha_phi_alpha)] = 0.0

    # unnormalized
    first_moment = mu_candidate - std_candidate * rho / c * (
        normal_pdf(beta) - normal_pdf(alpha)
    )

    second_moment = (
        std_candidate ** 2 * (1 - rho ** 2 / c) * (beta_phi_beta - alpha_phi_alpha)
        - mu_candidate ** 2
        + 2 * mu_candidate * first_moment
    )

    return second_moment ** 0.5


