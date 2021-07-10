import torch
from .utils import cdf

# THIS MES AQC
def MES(gp, testcase, thresholds, device, dtype):

    X_grid = testcase.X.to(device=device, dtype=dtype)

    # compute predictive posterior of Y(x) | train data
    likelihood = gp.likelihood
    gp.eval()
    likelihood.eval()

    # ok
    Y_pred_grid = likelihood(gp(X_grid))
    mean_tensor = Y_pred_grid.mean

    std_tensor = torch.sqrt(Y_pred_grid.variance)

    num_points = X_grid.size()[0]
    entropy_grid = torch.zeros(num_points,).to(device, dtype)

    for j in range(len(thresholds) - 1):
        p_j = cdf(mean_tensor, std_tensor, thresholds[j + 1]) \
                 - cdf(mean_tensor, std_tensor, thresholds[j])

        entropy_grid[p_j > 0] -= torch.log(torch.exp(p_j[p_j > 0])) \
                                    * torch.log(torch.exp(torch.log(p_j[p_j > 0])))
    return entropy_grid


def MES_test(gp, testcase, thresholds, X_grid, device, dtype):
    entropy_grid = torch.zeros(X_grid.size()[0],).to(device, dtype)
    for i, x in enumerate(X_grid):
        entropy_grid[i] = MES(gp, testcase, thresholds, x.view(1, -1), device, dtype)

    return entropy_grid
