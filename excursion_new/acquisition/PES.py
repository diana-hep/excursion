import torch
from .utils import truncated_std_conditional
from .utils import h_normal
from .base import AcquisitionFunction


class PES(AcquisitionFunction):
    def __init__(self, device=None, dtype=None, batch=False, ):
        self._prev_acq_point_index = []
        self.device = device
        self.dtype = dtype
        self.batch = batch
        self.grid = self.log = None

    def acquire(self, gp, thresholds, meshgrid):
        """
        Calculates information gain of choosing x_candidate as next point to evaluate.
        Performs this calculation with the Predictive Entropy Search approximation. Roughly,
        PES(x_candidate) = int dx { H[Y(x_candidate)] - E_{S(x=j)} H[Y(x_candidate)|S(x=j)] }
        Notation: PES(x_candidate) = int dx H0 - E_Sj H1

        """
        X_grid = torch.from_numpy(meshgrid).to(device=self.device, dtype=self.dtype)

        # compute predictive posterior of Y(x) | train data
        likelihood = gp.likelihood
        gp.eval()
        likelihood.eval()
        acquisition_values_grid = []
        thresholds = torch.Tensor(thresholds).to(device=self.device, dtype=self.dtype)

        for x_candidate in X_grid:
            x_candidate = x_candidate.view(1, -1).to(device=self.device, dtype=self.dtype)

            X_all = torch.cat((x_candidate, X_grid))  # .to(device, dtype)
            Y_pred_all = likelihood(gp(X_all))
            Y_pred_grid = torch.distributions.Normal(
                loc=Y_pred_all.mean[1:], scale=(Y_pred_all.variance[1:]) ** 0.5
            )

            # vector of expected value H1 under S(x) for each x in X_grid
            E_S_H1 = torch.zeros(len(X_grid)).to(device=self.device, dtype=self.dtype)

            for j in range(len(thresholds) - 1):
                # vector of sigma(Y(x_candidate)|S(x)=j) truncated
                trunc_std_j = truncated_std_conditional(
                    Y_pred_all, thresholds[j], thresholds[j + 1]
                )
                H1_j = h_normal(trunc_std_j)

                # vector of p(S(x)=j)
                p_j = Y_pred_grid.cdf(thresholds[j + 1]) - Y_pred_grid.cdf(thresholds[j])
                mask = torch.where(p_j == 0.0)
                H1_j[mask] = 0.0
                # print(p_j)
                # print(H1_j)
                E_S_H1 += p_j * H1_j  # expected value of H1 under S(x)

            # entropy of Y(x_candidate)
            H0 = h_normal(Y_pred_all.variance[0] ** 0.5)

            # info gain on the grid, vector
            info_gain = H0 - E_S_H1

            info_gain[~torch.isfinite(info_gain)] = 0.0  # just to avoid NaN

            # cumulative info gain over grid
            cumulative_info_gain = info_gain.sum()
            acquisition_values_grid.append(cumulative_info_gain)
        acquisition_values_grid = torch.Tensor(acquisition_values_grid).to(device=self.device, dtype=self.dtype)
        self.grid = acquisition_values_grid
        self.log = torch.clone(acquisition_values_grid)

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

