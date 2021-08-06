import numpy as np
import torch
# from excursion.utils import mgrid, mesh2points
import math


def truth(x):
    xv, yv = x[:, 0], x[:, 1]
    """more wiggles in physics case"""

    def xsec(xv, yv):
        return (
            12 * torch.exp(-xv / 2)
            + ((0.1 * torch.cos(10 * yv + 1)))
            + ((0.2 * torch.cos(15 * xv)))
        )

    def eff(xv, yv):
        return torch.tanh((1.3 * xv - yv) + 1)

    def stats(nevents):
        return (1 - torch.tanh((nevents - 5))) / 2.0

    def analysis(xv, yv):
        return stats(xsec(xv, yv) * eff(xv, yv))

    return 3 * (torch.log(analysis(xv, yv)) - math.log(0.05))

def test(x):
    xv, yv = x[:, 0], x[:, 1]

    return xv + yv


true_functions = [truth]
thresholds = [0.0]
bounding_box = [[0.0, 1.5], [0.0, 1.5]]
ndim = 2
plot_npoints = [10]*ndim
