import torch
import math
import numpy as np
# GET RID OF THIS, so leave the function as the module,


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


def numpy_func(x):
    xv, yv = x[:,0],x[:,1]
    '''more wiggles in physics case'''
    def xsec(xv,yv):
        return 12*np.exp(-xv/2)+((0.1*np.cos(10*yv+1)))+((0.2*np.cos(15*xv)))

    def eff(xv,yv):
        return np.tanh((1.3*xv-yv)+1)

    def stats(nevents):
        return (1-np.tanh((nevents-5)))/2.

    def analysis(xv,yv):
        return stats(xsec(xv,yv) * eff(xv,yv))

    return 3*(np.log(analysis(xv,yv)) - np.log(0.05))



def test(x):
    xv, yv = x[:, 0], x[:, 1]

    return xv**2 + yv**2

def true_function(X):
    if isinstance(X, torch.Tensor):
        return truth(X)
    else:
        return numpy_func(X)

# Put this into the intializer for the excursion problem
#
true_functions = [true_function]
test_functions = [test]

# thresholds = [0.0]
# bounding_box = [[0.0, 1.5], [0.0, 1.5]]
# ndim = 2
# grid_step_size = [41]*ndim
# acq_grid_npoints = grid_step_size