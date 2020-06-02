import gpytorch
import torch
from gpytorch.utils.broadcasting import _mul_broadcast_shape

class LinealMean(gpytorch.means.Mean):
    def __init__(self, prior=None, batch_shape=torch.Size(), **kwargs):
        super(LinealMean, self).__init__()
        self.batch_shape = batch_shape
        ndim=1
        self.register_parameter(name="slope", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, ndim)))
        self.register_parameter(name="constant", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        if prior is not None:
            self.register_prior("mean_prior", prior, "constant")

    def forward(self, inputs):
        if inputs.shape[:-2] == self.batch_shape:
            a_x = (self.slope * inputs).view(-1,)
            return self.constant.expand(inputs.shape[:-1]) + a_x.expand(inputs.shape[:-1])
        else:
        
            return self.slope.expand(_mul_broadcast_shape(inputs.shape[:-1], self.constant.shape))

            
# simplest form of GP model RBF with constant mean prior, exact inference
class ConstantMean(gpytorch.means.Mean):
    def __init__(self, prior=None, batch_shape=torch.Size(), **kwargs):
        super(ConstantMean, self).__init__()
        self.batch_shape = batch_shape
        self.register_parameter(name="constant", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        if prior is not None:
            self.register_prior("mean_prior", prior, "constant")

    def forward(self, inputs):
        if inputs.shape[:-2] == self.batch_shape:
            return self.constant.expand(inputs.shape[:-1])
        else:
            return self.constant.expand(_mul_broadcast_shape(inputs.shape[:-1], self.constant.shape))