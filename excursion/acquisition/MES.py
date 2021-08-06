import torch
from .utils import cdf
from .base import AcquisitionFunction


class MES(AcquisitionFunction):
    def __init__(self, device, dtype, batch=False, ):
        self._prev_acq_point_index = []
        self.device = device
        self.dtype = dtype
        self.batch = batch
        self.acq_vals = None

    def acquire(self, gp, thresholds, X_pointsgrid):
    # compute predictive posterior of Y(x) | trin data
        likelihood = gp.likelihood
        gp.eval()
        likelihood.eval()

    # ok
        prediction = likelihood(gp(X_pointsgrid))
        pred_mean = prediction.mean

        pred_stdev = torch.sqrt(prediction.variance)

        num_points = X_pointsgrid.size()[0]
        entropy_grid = torch.zeros(num_points,).to(device=self.device, dtype=self.dtype)

        for j in range(len(thresholds) - 1):
            p_j = cdf(pred_mean, pred_stdev, thresholds[j + 1]) - cdf(pred_mean, pred_stdev, thresholds[j])

            entropy_grid[p_j > 0] -= torch.log(torch.exp(p_j[p_j > 0])) * torch.log(torch.exp(torch.log(p_j[p_j > 0])))

        acq_cand_vals = entropy_grid
        self.acq_vals = torch.clone(entropy_grid)

        if self.batch:
            print("Will implement this later for batched results\n")
            pass

        return X_pointsgrid[self.get_first_max_index(gp, X_pointsgrid, acq_cand_vals)]

    def get_first_max_index(self, gp, X_pointsgrid, acq_cand_vals):
        X_train = gp.train_inputs[0].to(device=self.device, dtype=self.dtype)
        X_train = X_train.tolist()
        new_index = torch.argmax(acq_cand_vals)

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
            acq_cand_vals[new_index] = torch.Tensor([(-1.0) * float("Inf")])
            new_index = torch.argmax(acq_cand_vals)
            return self._check_prev_acq(new_index, X_train, X_pointsgrid, acq_cand_vals)