from excursion_new.utils import mgrid, mesh2points
import numpy as np
import torch


class ExcursionProblem(object):
    def __init__(self, functions, thresholds=[0.0], ndim=1, bounding_box=None, plot_npoints=None, init_n_points=2):
        self.functions = functions
        self.thresholds = thresholds
        self.bounding_box = np.asarray(bounding_box or [[0, 1]] * ndim)
        assert len(self.bounding_box) == ndim
        self.ndim = ndim
        plot_npoints = plot_npoints or [[101 if ndim < 3 else 31]] * ndim
        self.plot_rangedef = np.concatenate([self.bounding_box, np.asarray(plot_npoints).reshape(-1, 1)], axis=-1)
        self.plot_G = mgrid(self.plot_rangedef)
        self.plot_X = mesh2points(self.plot_G, self.plot_rangedef[:, 2])
        self.init_X_points = None
        self.init_n_points = init_n_points
        self.acq_func = None
        self.data_type = torch.float64

        # # For check_x_valid # #
        # def invalid_region(self, X):
        #     allvalid = lambda X: np.zeros_like(X[:, 0], dtype='bool')
        #     return self._invalid_region(X) if self._invalid_region else allvalid(X)

