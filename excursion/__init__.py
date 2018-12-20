from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import WhiteKernel
# from sklearn.gaussian_process.kernels import Matern
import logging
import time
import numpy as np
from . import utils

log = logging.getLogger(__name__)

def get_gp(X, y, alpha=10**-7, kernel_name='const_rbf'):
    start = time.time()
    if kernel_name == 'const_rbf':
        length_scale = [1.]*X.shape[-1]
        kernel = ConstantKernel() * RBF(length_scale_bounds=[0.1, 100.0], length_scale = length_scale)
    elif kernel_name == 'tworbf_white':
        kernel = ConstantKernel() * RBF(length_scale_bounds=[1e-2,100]) + \
                 ConstantKernel() * RBF(length_scale_bounds=[100., 1000.0]) + \
                 WhiteKernel(noise_level_bounds=[1e-7,1e-4])
    elif kernel_name == 'onerbf_white':
        kernel = ConstantKernel() * RBF(length_scale_bounds=[1e-2,100]) + WhiteKernel(noise_level_bounds=[1e-7,1e-1])
    else:
        raise RuntimeError('unknown kernel')
    gp = GaussianProcessRegressor(kernel=kernel,
                                  n_restarts_optimizer=10,
                                  alpha=alpha,
                                  random_state=1234)
    gp.fit(X, y.ravel())
    delta = time.time()-start
    log.info('made a GP for {} training points in {:.3f} seconds'.format(len(X),delta))
    return gp

class ExcursionProblem(object):
    def __init__(self, functions, thresholds = [0.0], ndim = 1, bounding_box = None, plot_npoints = None, invalid_region = None):
        self._invalid_region = invalid_region
        self.functions = functions
        self.thresholds = thresholds
        self.bounding_box = np.asarray(bounding_box or [[0,1]]*ndim)
        assert len(self.bounding_box) == ndim
        self.ndim = ndim
        plot_npoints = plot_npoints or [[101 if ndim < 3 else 31]]*ndim
        self.plot_rangedef = np.concatenate([self.bounding_box,np.asarray(plot_npoints).reshape(-1,1)],axis=-1)
        self.plotG = utils.mgrid(self.plot_rangedef)
        self.plotX = utils.mesh2points(self.plotG,self.plot_rangedef[:,2])

    def invalid_region(self,X):
        allvalid = lambda X: np.zeros_like(X[:,0], dtype = 'bool')
        return self._invalid_region(X) if self._invalid_region else allvalid(X)

    def random_points(self,N, seed = None):
        np.random.seed(seed)
        in_bounding = np.random.uniform(
            self.bounding_box[:,0],
            self.bounding_box[:,1],
            size = (N if not self._invalid_region else N*50,self.ndim)
        )
        if not self._invalid_region: return in_bounding
        valid = in_bounding[~self.invalid_region(in_bounding)]
        idx = np.random.choice(np.arange(0,len(valid)),size = N)
        points = valid[idx]
        assert len(points) == N
        return points

    def acqX(self):
        return self.random_points(500)
        
    def meanX(self):
        return self.random_points(500)
