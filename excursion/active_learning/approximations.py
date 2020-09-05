from scipy.linalg import cho_solve
from scipy.stats import norm
import gpytorch
import torch
import numpy as np

torch.cuda.set_device(0)


def h_normal_gpytorch(s):
    """ Entropy of a normal distribution """
    return torch.log(s * (2 * np.e * np.pi) ** 0.5)


def approx_mi_vec_gpytorch(mu, cov, thresholds):
    # Expectation Propagation
    mu1 = mu[:, 0]
    std1 = cov[:, 0, 0] ** 0.5
    mu2 = mu[:, 1]
    std2 = cov[:, 1, 1] ** 0.5
    rho = cov[:, 0, 1] / (std1 * std2)

    std_sx = []

    for j in range(len(thresholds) - 1):
        alpha_j = (thresholds[j] - mu2) / std2
        beta_j = (thresholds[j + 1] - mu2) / std2
        alpha_j = alpha_j.detach().numpy()
        beta_j = beta_j.detach().numpy()

        c_j = norm.cdf(beta_j) - norm.cdf(alpha_j)

        # \sigma(Y(X)|S(x')=j)
        b_phi_b = beta_j * norm.pdf(beta_j)
        b_phi_b[~np.isfinite(beta_j)] = 0.0
        a_phi_a = alpha_j * norm.pdf(alpha_j)
        a_phi_a[~np.isfinite(alpha_j)] = 0.0

        alpha_j = torch.tensor(alpha_j)
        beta_j = torch.tensor(beta_j)

        mu_cond = mu1 - std1 * rho / torch.tensor(c_j) * (
            torch.tensor(norm.pdf(beta_j)) - torch.tensor(norm.pdf(alpha_j))
        )
        var_cond = (
            mu1 ** 2
            - 2
            * mu1
            * std1
            * (
                rho
                / torch.tensor(c_j)
                * (torch.tensor(norm.pdf(beta_j)) - torch.tensor(norm.pdf(alpha_j)))
            )
            + std1 ** 2
            * (
                1.0
                - (rho ** 2 / torch.tensor(c_j))
                * (torch.tensor(b_phi_b) - torch.tensor(a_phi_a))
            )
            - mu_cond ** 2
        )

        std_sx_j = var_cond ** 0.5

        std_sx.append(std_sx_j)

    # Entropy
    h = h_normal_gpytorch(std1)

    for j in range(len(thresholds) - 1):
        p_j = norm(mu2.detach().numpy(), std2.detach().numpy()).cdf(
            thresholds[j + 1]
        ) - norm(mu2.detach().numpy(), std2.detach().numpy()).cdf(thresholds[j])
        print("pj ", p_j)
        dec = torch.tensor(p_j) * h_normal_gpytorch(std_sx[j])
        h[p_j > 0.0] -= dec[p_j > 0.0]

    return h
