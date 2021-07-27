import torch
from .utils import cdf
from .base import AcquisitionFunction


class MES(AcquisitionFunction):
    def __init__(self, device, dtype, batch=False, ):
        self._prev_acq_point_index = []
        self.device = device
        self.dtype = dtype
        self.batch = batch
        self.grid = self.log = None

    def acquire(self, gp, thresholds, meshgrid):
        X_grid = torch.from_numpy(meshgrid).to(device=self.device, dtype=self.dtype)

    # compute predictive posterior of Y(x) | trin data
        likelihood = gp.likelihood
        gp.eval()
        likelihood.eval()

    # ok
        Y_pred_grid = likelihood(gp(X_grid))
        mean_tensor = Y_pred_grid.mean

        std_tensor = torch.sqrt(Y_pred_grid.variance)

        num_points = X_grid.size()[0]
        entropy_grid = torch.zeros(num_points,).to(device=self.device, dtype=self.dtype)

        for j in range(len(thresholds) - 1):
            p_j = cdf(mean_tensor, std_tensor, thresholds[j + 1]) \
                  - cdf(mean_tensor, std_tensor, thresholds[j])

            entropy_grid[p_j > 0] -= torch.log(torch.exp(p_j[p_j > 0])) \
                                      * torch.log(torch.exp(torch.log(p_j[p_j > 0])))

        self.grid = entropy_grid
        self.log = torch.clone(entropy_grid)

        if self.batch:
            print("Will implement this later for batched results\n")
            pass

        return X_grid[self.get_first_max_index(gp, X_grid)]

    def get_first_max_index(self, gp, meshgrid):
        X_train = gp.train_inputs[0].to(device=self.device, dtype=self.dtype)
        X_train = X_train.tolist()
        new_index = torch.argmax(self.grid)

        # if the index is not already picked nor in the training set
        # accept it ans remove from future picks
        return self._check_prev_acq(new_index, X_train, meshgrid)

    def _check_prev_acq(self, new_index, X_train, meshgrid):
        new_X = meshgrid[new_index]
        if (new_index not in self._prev_acq_point_index) and (
                new_X.tolist() not in X_train):
            self.pop(new_index)
            self._prev_acq_point_index.append(new_index.item())
            return new_index.item()
        else:
            self.pop(new_index)
            new_index = torch.argmax(self.grid)
            return self._check_prev_acq(new_index, X_train, meshgrid)

    def pop(self, index):
            self.grid[index] = torch.Tensor([(-1.0) * float("Inf")])

