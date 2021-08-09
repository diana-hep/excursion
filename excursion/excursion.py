from .utils import mgrid, mesh2points
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


class ExcursionProblem(object):
    def __init__(self, functions, thresholds=[0.5], ndim=1, bounding_box=None, grid_step_size=None, init_n_points=None):
        self.functions = functions
        self.thresholds = thresholds
        self.bounding_box = np.asarray(bounding_box or [[0, 1]] * ndim)
        assert len(self.bounding_box) == ndim
        self.ndim = ndim
        grid_step_size = grid_step_size or [[101 if ndim < 3 else 31]] * ndim
        self.rangedef = np.concatenate([self.bounding_box, np.asarray(grid_step_size).reshape(-1, 1)], axis=-1)
        self.X_meshgrid = mgrid(self.rangedef)
        self.X_pointsgrid = mesh2points(self.X_meshgrid, self.rangedef[:, 2])
        self.acq_pointsgrid = self.X_pointsgrid
        self.init_n_points = init_n_points # future place to store init points that were a snapshot of past training
        self.dtype = torch.float64
        self._invalid_region = None


    def invalid_region(self, X):
        allvalid = lambda X: np.zeros_like(X[:, 0], dtype='bool')
        return self._invalid_region(X) if self._invalid_region else allvalid(X)


# # move this into the excursion result, unless we add scikit learn implementation # #

def build_result(details: ExcursionProblem, acquisition, **kwargs):
    X_pointsgrid = torch.from_numpy(details.X_pointsgrid).to(device=kwargs['device'], dtype=details.dtype)
    true_y = details.functions[0](X_pointsgrid).cpu().detach().numpy()
    acquisition = acquisition # What if they passed in their own acq, then there is no string here.
    return ExcursionResult(ndim=details.ndim, thresholds=details.thresholds, true_y=true_y,
                           invalid_region=details.invalid_region, X_pointsgrid=details.X_pointsgrid,
                           X_meshgrid=details.X_meshgrid, rangedef=details.rangedef)


class ExcursionResult(object):

    def __init__(self, ndim, thresholds, true_y, invalid_region, X_pointsgrid, X_meshgrid, rangedef, acq_vals=[], train_X=[],
                 train_y=[], pred_mean=[], pred_cov=[], next_x=[]):

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
        self.acq_vals = acq_vals
        self.mean = pred_mean
        self.cov = pred_cov
        self.next_x = next_x
        self.train_X = train_X
        self.train_y = train_y
        self.confusion_matrix = []
        self.pct_correct = []

    def update(self, model, next_x, acq_vals, X_pointsgrid, log=True):
        train_X = model.train_inputs[0].cpu().detach().numpy()
        train_y = model.train_targets.cpu().detach().numpy()
        likelihood = model.likelihood
        likelihood.eval()
        model.eval()
        prediction = likelihood(model(X_pointsgrid))
        variance = prediction.variance.cpu().detach().numpy()
        mean = prediction.mean.cpu().detach().numpy()
        if acq_vals is not None:
            acq_vals = acq_vals.cpu().detach().numpy()
        if next_x is not None:
            next_x = next_x.cpu().detach().numpy()
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
        print("Accuracy %", pct_correct)
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
