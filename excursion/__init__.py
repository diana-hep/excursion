import logging
import time
import numpy as np
from . import utils
from . import optimize
from . import diagnosis
from .samplers import latin_sample_n

log = logging.getLogger(__name__)

class ExcursionProblem(object):
    def __init__(self, functions, thresholds = [0.0], ndim = 1, bounding_box = None, plot_npoints = None, invalid_region = None, testdata = None, n_acq = 2000, n_mean = 2000):
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
        self._testdata = testdata
        self._nmean = n_acq
        self._nacq = n_mean
    
    def testdata(self):
        if self._testdata:
            return self._testdata
        testX = self.plotX[~self.invalid_region(self.plotX)]
        testy_list = [func(testX) for func in self.functions]
        testdata = testX, testy_list
        return testdata

    def invalid_region(self,X):
        allvalid = lambda X: np.zeros_like(X[:,0], dtype = 'bool')
        return self._invalid_region(X) if self._invalid_region else allvalid(X)

    def random_points(self,N, seed = None):
        np.random.seed(seed)
        return latin_sample_n(self, N, self.ndim)

    def acqX(self):
        return self.random_points(self._nacq)
        
    def meanX(self):
        return self.random_points(self._nmean)

class Learner(object):
    def __init__(self, scandetails, gp_maker =  optimize.get_gp, evaluator = optimize.default_evaluator):
        self.scandetails = scandetails
        self.gp_maker = gp_maker
        self.evaluator = evaluator
        self.metrics = []
        self.X = np.empty((0,scandetails.ndim))
        self.y_list = [np.empty((0,)) for f in scandetails.functions ]

    def evaluate_metrics(self):
        return diagnosis.diagnose(self.X,self.y_list,self.gps, self.scandetails)

    def initialize(self,n_init = 5, seed = None, snapshot = None):
        if not snapshot:
            self.X, self.y_list, self.gps = optimize.init(
                self.scandetails, n_init, seed, self.evaluator,self.gp_maker
            )
            self.metrics.append(self.evaluate_metrics())
        else:
            self.X = np.asarray(snapshot['X'])
            self.y_list = [np.asarray(y) for y in snapshot['y_list']]
            self.gps = [self.gp_maker(self.X,yl) for yl in self.y_list]
            self.metrics = snapshot['metrics']

    def suggest(self, batchsize = 1, resampling_frac = 0.30):
        return optimize.suggest(
            self.gps, self.X, self.scandetails,
            gp_maker = self.gp_maker, batchsize=batchsize,
            resampling_frac = resampling_frac, return_acqvals = False
        )

    def tell(self, newX, newys_list):
        self.X,self.y_list,self.gps = optimize.tell(
            self.X, self.y_list, self.scandetails,
            newX, newys_list, self.gp_maker
        )
        self.metrics.append(self.evaluate_metrics())
        
    def evaluate_and_tell(self,newX):
        newys_list = self.evaluator(self.scandetails,newX)
        self.tell(newX,newys_list)

