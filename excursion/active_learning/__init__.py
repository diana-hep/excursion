import datetime
import torch
import gpytorch
import numpy as np
import math
from scipy.stats import norm
from scipy.linalg import cho_solve
from excursion.utils import h_normal
from torch.distributions import Normal
from excursion.utils import truncated_std_conditional
import time
import os
import gc



#### THIS CDF FUNCTION

def cdf(mu, sigma, t):
    # use torch.erfc for numerical stability
    erf = torch.erf((t - mu) * sigma.reciprocal() / math.sqrt(2))
    cdf = 0.5 * (1 + erf)
    return cdf


def MES_test(gp, testcase, thresholds, X_grid, device, dtype):
    entropy_grid = torch.zeros(X_grid.size()[0],).to(device, dtype)
    for i, x in enumerate(X_grid):
        entropy_grid[i] = MES(gp, testcase, thresholds, x.view(1, -1), device, dtype)

    return entropy_grid


#### THIS MES AQC

def MES_gpu(gp, testcase, thresholds, X_grid, device, dtype):

    # compute predictive posterior of Y(x) | train data
    likelihood = gp.likelihood
    gp.eval()
    likelihood.eval()

    # ok
    Y_pred_grid = likelihood(gp(X_grid))
    #print(Y_pred_grid)
    mean_tensor = Y_pred_grid.mean

    # print('Y_pred_grid.lazy_covariance_matrix.kernel', Y_pred_grid.lazy_covariance_matrix.kernel )
    # print('Y_pred_grid.lazy_covar_matrix.diag ', Y_pred_grid.lazy_tensors.size())

    std_tensor = torch.sqrt(Y_pred_grid.variance)
    #print(std_tensor.size())
    # std_tensor = torch.sqrt(torch.diag(Y_pred_grid.lazy_covariance_matrix))
    # print('mean_tensor ', mean_tensor.size())
    # print('std_tensor ', std_tensor.size())

    num_points = X_grid.size()[0]
    entropy_grid = torch.zeros(num_points,).to(device, dtype)

    for j in range(len(thresholds) - 1):
        my_p_j = cdf(mean_tensor, std_tensor, thresholds[j + 1]) \
                 - cdf(mean_tensor, std_tensor, thresholds[j])

        # print(my_p_j[my_p_j > 0].size())
        # print('sumexp_pj_matrix=', torch.log(torch.exp(my_p_j[my_p_j > 0])).tolist())
        # print(my_p_j)
        entropy_grid[my_p_j > 0] -= torch.log(torch.exp(my_p_j[my_p_j > 0])) \
                                    * torch.log(torch.exp(torch.log(my_p_j[my_p_j > 0])))
        # print(entropy_grid[my_p_j > 0])
        # print(entropy_grid)

        # test with MES

        # my_pj_vector = []
        # pj_vector = []
        # sumexp_pj_good = []

        # for i, x in enumerate(X_grid):
        #    #print('***x candidate ', x)
        #    Y_pred_candidate = likelihood(gp(x))
        #    normal_candidate = torch.distributions.Normal(
        #    loc=Y_pred_candidate.mean, scale=Y_pred_candidate.variance ** 0.5
        #    )

        #    p_j = normal_candidate.cdf(thresholds[j + 1]) - normal_candidate.cdf(thresholds[j])
        #    my_p_j = cdf(normal_candidate.mean, normal_candidate.variance ** 0.5, thresholds[j + 1]
        #    ) - cdf(normal_candidate.mean, normal_candidate.variance ** 0.5, thresholds[j])

        #    pj_vector.append(p_j.item())
        #    my_pj_vector.append(my_p_j.item())

        #    if(p_j>0):
        #        sumexp_pj_good.append( torch.logsumexp(p_j, 0).item() )

        # print('sumexp_pj_good=', sumexp_pj_good)
        # print('sumexp_pj_my=', torch.logsumexp(my_p_j, 0))

    return entropy_grid


def MES(gp, testcase, thresholds, x_candidate, device, dtype):

    # compute predictive posterior of Y(x) | train data
    likelihood = gp.likelihood
    gp.eval()
    likelihood.eval()

    Y_pred_candidate = likelihood(gp(x_candidate))

    normal_candidate = torch.distributions.Normal(
        loc=Y_pred_candidate.mean, scale=Y_pred_candidate.variance ** 0.5
    )

    # entropy of S(x_candidate)
    entropy_candidate = torch.Tensor([0.0]).to(device, dtype)

    for j in range(len(thresholds) - 1):
        # p(S(x)=j)
        p_j = normal_candidate.cdf(thresholds[j + 1]) - normal_candidate.cdf(
            thresholds[j]
        )
        my_p_j = cdf(
            normal_candidate.mean, normal_candidate.variance ** 2, thresholds[j + 1]
        ) - cdf(normal_candidate.mean, normal_candidate.variance ** 2, thresholds[j])

        if p_j > 0.0:
            # print(x_candidate, p_j,j)

            entropy_candidate -= torch.logsumexp(p_j, 0).to(
                device, dtype
            ) * torch.logsumexp(torch.log(p_j), 0)

    return entropy_candidate.detach()  # .to(device, dtype)


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
        E_S_H1 += p_j * H1_j  # expected value of H1 under S(x)

    # entropy of Y(x_candidate)
    H0 = h_normal(Y_pred_all.variance[0] ** 0.5)

    # info gain on the grid, vector
    info_gain = H0 - E_S_H1

    info_gain[~torch.isfinite(info_gain)] = 0.0  # just to avoid NaN

    # cumulative info gain over grid
    cumulative_info_gain = info_gain.sum()

    return cumulative_info_gain.item()


def PPES(gp, testcase, thresholds, x_candidate):
    """
    Calculates information gain of choosing x_candidadate as next point to evaluate.
    Performs this calculation with the Predictive Entropy Search approximation weighted by the posterior. 
    Roughly,
    PES(x_candidate) = int Y(x)dx { H[Y(x_candidate)] - E_{S(x=j)} H[Y(x_candidate)|S(x=j)] }
    Notation: PES(x_candidate) = int dx H0 - E_Sj H1

    """

    # compute predictive posterior of Y(x) | train data
    raise NotImplmentedError(
        "Should be same strcture as PES but the cumulative info gain is weighted"
    )


acquisition_functions = {
    "PES": PES,
    "MES": MES,
    "MES_gpu": MES_gpu,
    "MES_test": MES_test,
}
