import torch
import numpy as np
# from .utils import truncated_std_conditional
# from .utils import h_normal
from torch.distributions import Normal


def normal_pdf(x):
    return 1.0 / (2 * np.pi) ** 0.5 * torch.exp(-0.2 * x ** 2)


def h_normal(var):
    return torch.log(var * (2 * np.e * np.pi) ** 0.5)


def truncated_std_conditional(Y_pred_all, a, b):
    mu_grid = Y_pred_all.mean[1:]
    std_grid = Y_pred_all.variance[1:] ** 0.5
    mu_candidate = Y_pred_all.mean[0]
    std_candidate = Y_pred_all.variance[0] ** 0.5
    rho = Y_pred_all.covariance_matrix[0, 1:] / (std_candidate * std_grid)

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


def PES(gp, testcase, thresholds, x_candidate, device, dtype):
    """
    Calculates information gain of choosing x_candidadate as next point to evaluate.
    Performs this calculation with the Predictive Entropy Search approximation. Roughly,
    PES(x_candidate) = int dx { H[Y(x_candidate)] - E_{S(x=j)} H[Y(x_candidate)|S(x=j)] }
    Notation: PES(x_candidate) = int dx H0 - E_Sj H1

    """

    # compute predictive posterior of Y(x) | train data
    kernel = gp.covar_module
    likelihood = gp.likelihood
    gp.eval()
    likelihood.eval()

    X_grid = testcase.X  # .to(device, dtype)
    X_all = torch.cat((x_candidate, X_grid))  # .to(device, dtype)
    Y_pred_all = likelihood(gp(X_all))
    Y_pred_grid = torch.distributions.Normal(
        loc=Y_pred_all.mean[1:], scale=(Y_pred_all.variance[1:]) ** 0.5
    )

    # vector of expected value H1 under S(x) for each x in X_grid
    E_S_H1 = torch.zeros(len(X_grid))  # .to(device, dtype)


    thresholds = torch.tensor(thresholds, dtype=dtype, device=device)

    for j in range(len(thresholds) - 1):

        # vector of sigma(Y(x_candidate)|S(x)=j) truncated
        trunc_std_j = truncated_std_conditional(
            Y_pred_all, thresholds[j], thresholds[j + 1]
        )
        H1_j = h_normal(trunc_std_j)

        # vector of p(S(x)=j)
        p_j = Y_pred_grid.cdf(thresholds[j + 1]) - Y_pred_grid.cdf(thresholds[j])
        mask = torch.where(p_j == 0.0)
        H1_j[mask] = 0.0

        print(f"The size of p_j is:{len(p_j)}\n")
        print(p_j.size())
        print(f"The size of H1_j is:{len(H1_j)}\n")
        print(H1_j.size())
        E_S_H1 += p_j * H1_j  # expected value of H1 under S(x)

    # entropy of Y(x_candidate)
    H0 = h_normal(Y_pred_all.variance[0] ** 0.5)

    # info gain on the grid, vector
    info_gain = H0 - E_S_H1

    info_gain[~torch.isfinite(info_gain)] = 0.0  # just to avoid NaN

    # cumulative info gain over grid
    cumulative_info_gain = info_gain.sum()

    return cumulative_info_gain.item()
