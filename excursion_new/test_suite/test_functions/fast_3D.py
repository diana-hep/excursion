import numpy as np
import torch
import math
# from excursion.utils import mgrid, mesh2points


def truth(x):
    xv, yv, zv = x[:, 0], x[:, 1], x[:, 2]
    """more wiggles in physics case"""

    def xsec(xv, yv, zv):
        return (
            12 * torch.exp(-xv / 2)
            + ((0.1 * torch.cos(10 * yv)))
            + ((0.2 * torch.cos(15 * xv)))
        ) * torch.exp(-0.3 * zv)

    def eff(xv, yv, zv):
        return torch.tanh((1.3 * xv - yv) + 1) * 1

    def stats(nevents):
        return (1 - torch.tanh((nevents - 5))) / 2.0

    def analysis(xv, yv, zv):
        return stats(xsec(xv, yv, zv) * eff(xv, yv, zv))

    return 3 * (torch.log(analysis(xv, yv, zv)) - math.log(0.05))

true_functions = [truth]

# Define threshold list
thresholds = [0.0]
bounding_box = [[0.0, 1.5], [0.0, 1.5], [0, 1.5]]
# Define grid for acquisition function
ndim = 3
plot_npoints = [41]*ndim

