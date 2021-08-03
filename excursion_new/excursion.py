from excursion_new.utils import mgrid, mesh2points
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


class ExcursionProblem(object):
    def __init__(self, functions, thresholds=[0.5], ndim=1, bounding_box=None, plot_npoints=None, init_n_points=2):
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
        self.dtype = torch.float64
        self._invalid_region = None
        # # For check_x_valid # #

    def invalid_region(self, X):
        allvalid = lambda X: np.zeros_like(X[:, 0], dtype='bool')
        return self._invalid_region(X) if self._invalid_region else allvalid(X)


# # move this into the excursion result, unless we add scikit learn implementation # #

def build_result(details: ExcursionProblem, model, acquisition, next_x, **kwargs):
    train_X = model.train_inputs[0].cpu().detach().numpy()
    train_y = model.train_targets.cpu().detach().numpy()
    plot_X = torch.from_numpy(details.plot_X).to(device=kwargs['device'], dtype=kwargs['dtype'])
    likelihood = model.likelihood
    likelihood.eval()
    model.eval()
    prediction = likelihood(model(plot_X))
    variance = prediction.variance.cpu().detach().numpy()
    mean = prediction.mean.cpu().detach().numpy()
    true_y = details.functions[0](plot_X).cpu().detach().numpy()
    if acquisition is not None:
        acquisition = acquisition.cpu().detach().numpy()
    if next_x is not None:
        next_x = next_x.cpu().detach().numpy()
    return ExcursionResult(ndim=details.ndim, acquisition=acquisition, train_X=train_X, train_y=train_y,
                           plot_X=details.plot_X, plot_G=details.plot_G, rangedef=details.plot_rangedef,
                           pred_mean=mean, pred_cov=variance, thresholds=details.thresholds, next_x=next_x,
                           true_y=true_y, invalid_region=details.invalid_region)


class ExcursionResult(object):

    def __init__(self, ndim, acquisition, train_X, train_y, plot_X, plot_G, rangedef,
                 pred_mean, pred_cov, thresholds, next_x, true_y, invalid_region):
        self.acq = acquisition
        self.train_X = train_X
        self.train_y = train_y
        self.plot_X = plot_X
        self.plot_G = plot_G
        self.rangedef = rangedef
        self.mean = pred_mean
        self.cov = pred_cov
        self.thr = thresholds
        self.next_x = next_x
        self.true_y = true_y
        self.invalid_region = invalid_region
        self.ndim = ndim
        self.confusion_matrix = self.get_confusion_matrix()
        self.pct_correct = self.get_percent_correct()

    def get_percent_correct(self):
        pct_correct = np.diag(self.confusion_matrix).sum() * 1.0 / len(self.plot_X)
        print("Accuracy %", pct_correct)
        return pct_correct

    def get_confusion_matrix(self):
        thresholds = [-np.inf] + self.thr + [np.inf]

        def label(y):
            for j in range(len(thresholds) - 1):
                if thresholds[j + 1] > y >= thresholds[j]:
                    return int(j)

        labels_pred = np.array([label(y) for y in self.mean])
        isnan_vector = np.isnan(labels_pred)
        labels_true = np.array([label(y) for y in self.true_y])
        return confusion_matrix(labels_true, labels_pred)
