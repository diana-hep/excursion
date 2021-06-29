from gpytorch.kernels import *
from gpytorch.constraints import GreaterThan


class Kernel(object):

    # Extend options to **kwargs
    def __init__(self, model_type: str, base_kernel: str, grid = None):
    #    try:
        if grid is None:
            self.kernel = globals()[model_type](globals()[base_kernel](
                lengthscale_constraint=GreaterThan(lower_bound=0.1)))
        else:
            self.kernel = globals()[model_type](globals()[base_kernel](),
                                                grid=grid)
    #    except:

    def get_kernel(self):
        return self.kernel
