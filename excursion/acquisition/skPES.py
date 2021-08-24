import gc

from .base import AcquisitionFunction
import numpy as np
import os
from . import utils


class SKPES(AcquisitionFunction):
    def __init__(self, device=None, dtype=None, batch=False, ):
        self._prev_acq_point_index = []
        self.device = device
        self.dtype = dtype
        self.batch = batch
        self.acq_vals = None

    def _acquire(self, gp, thresholds, X_pointsgrid, x_candidate):
        try:
            from joblib import Parallel, delayed
            print("Parallel Acq")
            nparallel = int(os.environ.get('EXCURSION_NPARALLEL', os.cpu_count()))
            result = Parallel(nparallel)(
                delayed(utils.info_gain)(x_candidate, [gp], thresholds, X_pointsgrid) for x_candidate in X_pointsgrid)
            gc.collect()
            return np.asarray(result)
        except ImportError:
            return np.array([utils.info_gain(x_candidate, [gp], thresholds, X_pointsgrid) for x_candidate in X_pointsgrid])

    def acquire(self, gp, thresholds, X_pointsgrid):
        """
        Calculates information gain of choosing x_candidate as next point to evaluate.
        Performs this calculation with the Predictive Entropy Search approximation. Roughly,
        PES(x_candidate) = int dx { H[Y(x_candidate)] - E_{S(x=j)} H[Y(x_candidate)|S(x=j)] }
        Notation: PES(x_candidate) = int dx H0 - E_Sj H1

        """

        acquisition_values = self._acquire(gp, thresholds, X_pointsgrid, None)
        self.acq_vals = np.copy(acquisition_values)

        X_train = gp.X_train_.tolist()

        for i, cacq in enumerate(X_pointsgrid[np.argsort(acquisition_values)]):
            if cacq.tolist() not in X_train:
                newx = cacq
                return newx

        return X_pointsgrid[self.get_first_max_index(gp, X_pointsgrid, acquisition_values)]


    def get_first_max_index(self, gp, X_pointsgrid, acq_cand_vals):
        X_train = gp.X_train_.tolist()
        new_index = np.argmax(acq_cand_vals)
        print(acq_cand_vals, new_index)

        # if the index is not already picked nor in the training set
        # accept it ans remove from future picks
        return self._check_prev_acq(new_index, X_train, X_pointsgrid, acq_cand_vals)

    # recursion helper function
    def _check_prev_acq(self, new_index, X_train, X_pointsgrid, acq_cand_vals):
        new_X = X_pointsgrid[new_index]
        if (new_index not in self._prev_acq_point_index) and (
                new_X.tolist() not in X_train):
            self._prev_acq_point_index.append(new_index.item())
            return new_index.item()
        else:
            acq_cand_vals[new_index] = np.array([(-1.0) * float("Inf")])
            # new_index = np.argmax(acq_cand_vals)
            return self._check_prev_acq(np.argmax(acq_cand_vals), X_train, X_pointsgrid, acq_cand_vals)