from .utils import mgrid, mesh2points
import numpy as np
from sklearn.metrics import confusion_matrix


class ExcursionProblem(object):
    def __init__(self, functions, thresholds=[0.5], ndim=1, bounding_box=None, grid_step_size=None, init_n_points=None):
        self.functions = functions
        self.thresholds = thresholds
        self.bounding_box = np.asarray(bounding_box or [[0, 1]] * ndim)
        assert len(self.bounding_box) == ndim
        self.ndim = ndim
        grid_step_size = grid_step_size or [[41 if ndim < 3 else 31]] * ndim
        self.rangedef = np.concatenate([self.bounding_box, np.asarray(grid_step_size).reshape(-1, 1)], axis=-1)
        self.X_meshgrid = mgrid(self.rangedef)
        self.X_pointsgrid = mesh2points(self.X_meshgrid, self.rangedef[:, 2])
        self.acq_pointsgrid = self.X_pointsgrid
        self.init_n_points = init_n_points # future place to store init points that were a snapshot of past training
        self._invalid_region = None

    def invalid_region(self, X):
        allvalid = lambda X: np.zeros_like(X[:, 0], dtype='bool')
        return self._invalid_region(X) if self._invalid_region else allvalid(X)


class ExcursionResult(object):

    def __init__(self, ndim, thresholds, true_y, invalid_region, X_pointsgrid, X_meshgrid, rangedef,
                 acq_vals=None, mean=None, cov=None, next_x=None, train_X=None, train_y=None):

        # need to do acq vals and acq grids
        # acq x and acq values
        self.true_y = true_y
        self.invalid_region = invalid_region
        # plot meshgrid and plot_x_points
        self.X_pointsgrid = X_pointsgrid
        self.X_meshgrid = X_meshgrid
        self.thresholds = thresholds
        self.rangedef = rangedef
        self.ndim = ndim

        # To be updated if log is true
        self.acq_vals = [] if acq_vals is None else [acq_vals]
        self.mean = [] if mean is None else [mean]
        self.cov = [] if cov is None else [cov]
        self.next_x = [] if next_x is None else [next_x]
        self.train_X = [] if train_X is None else [train_X]
        self.train_y = [] if train_y is None else [train_y]
        self.confusion_matrix = []
        self.pct_correct = []

    def update_result(self, model, next_x, acq_vals, X_pointsgrid, log=True):
        if model.device == 'skcpu':
            train_X = model.X_train_
            train_y = model.y_train_
            mean, variance = model.predict(X_pointsgrid, return_std=True)
        else:
            raise NotImplementedError("Only supports device type 'skcpu'")
        if log:
            self.acq_vals.append(acq_vals)
            self.mean.append(mean)
            self.cov.append(variance)
            self.next_x.append(next_x)
            self.train_X.append(train_X)
            self.train_y.append(train_y)
            self.get_confusion_matrix()
            self.get_percent_correct()
        else:
            self.acq_vals = [acq_vals]
            self.mean = [mean]
            self.cov = [variance]
            self.next_x = [next_x]
            self.train_X = [train_X]
            self.train_y = [train_y]
            self.get_confusion_matrix()
            self.get_percent_correct()

    def get_diagnostic(self):
        return self.confusion_matrix[-1], self.pct_correct[-1]

    def get_percent_correct(self):
        pct_correct = np.diag(self.confusion_matrix[-1]).sum() * 1.0 / len(self.X_pointsgrid)
        # print("Accuracy %", pct_correct)
        self.pct_correct.append(pct_correct)
        return self.pct_correct[-1]

    def get_confusion_matrix(self):
        thresholds = [-np.inf] + self.thresholds + [np.inf]

        def label(y):
            for j in range(len(thresholds) - 1):
                if thresholds[j + 1] > y >= thresholds[j]:
                    return int(j)

        labels_pred = np.array([label(y) for y in self.mean[-1]])
        isnan_vector = np.isnan(labels_pred)
        labels_true = np.array([label(y) for y in self.true_y])
        self.confusion_matrix.append(confusion_matrix(labels_true, labels_pred))
        return self.confusion_matrix[-1]

    def get_last_result(self):
        if not self.train_y:
            return ExcursionResult(ndim=self.ndim, thresholds=self.thresholds, true_y=self.true_y,
                           invalid_region=self.invalid_region, X_pointsgrid=self.X_pointsgrid,
                           X_meshgrid=self.X_meshgrid, rangedef=self.rangedef)
        else:
            return ExcursionResult(acq_vals=self.result.acq_vals[-1], train_y=self.result.train_y[-1],
                               train_X=self.result.train_X[-1], plot_X=self.result.X_pointsgrid,
                               plot_G=self.result.X_meshgrid, rangedef=self.result.rangedef,
                               pred_mean=self.result.mean[-1], pred_cov=self.result.cov[-1],
                               thresholds=self.result.thresholds, next_x=self.result.next_x[-1],
                               true_y=self.result.true_y[-1], invalid_region=self.result.invalid_region)
