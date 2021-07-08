import torch
import gpytorch
import os
import itertools
from excursion import get_gp, fit_hyperparams
    
from copy import deepcopy


class batchGrid(object):
    """
    A class to represent the underlying grid with useful features for batch point selection

    ...

    Attributes
    -----------
    batch_types : dict()
    grid : torch.Tensor
        the acquisition values for each point in the grid
    device : str
        device to choose grom gou or cpu
    picked_indexs list
        list to keep track of the indices already selected for query or the batch
    _ndims :  int
        dimension of the grid


    Methods
    -------
    pop(index)
        Removes index from to avoid being picked again
    update(acq_value_grid)
        Actualize the elements of the acquisition values for the same grid
    
    
    """

    def __init__(self, acq_values_of_grid, device, dtype, n_dims):
        self.grid = torch.as_tensor(acq_values_of_grid, device=device, dtype=dtype)
        self.batch_types = {
            "Naive": self.get_naive_batch,
            "KB": self.get_kb_batch,
            "Distanced": self.get_distanced_batch,
            # "Cluster": self.get_cluster_batch,
        }
        self.picked_indexs = []
        self._n_dims = n_dims
        self.device = device
        self.dtype = dtype

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        args = [a._t if hasattr(a, "_t") else a for a in args]
        ret = func(*args, **kwargs)
        return batchgrid(ret, kwargs["device"], kwargs["dtype"])

    def pop(self, index):
        self.grid[index] = torch.Tensor([(-1.0) * float("Inf")])

    def update(self, acq_values_of_grid, device, dtype):
        self.grid = torch.as_tensor(acq_values_of_grid, device=device, dtype=dtype)

    def get_first_max_index(self, gp, testcase, device, dtype):
        X_train = gp.train_inputs[0].to(device, dtype)

        new_index = torch.argmax(self.grid)
        new_x = testcase.X.to(device, dtype)[new_index]

        # if the index is not already picked nor in the training set
        # accept it ans remove from future picks
        if (new_index not in self.picked_indexs) and (
            new_x.tolist() not in X_train.tolist()
        ):
            self.pop(new_index)
            self.picked_indexs.append(new_index.item())
            return new_index.item()

        else:
            self.pop(new_index)
            return self.get_first_max_index(gp, testcase, device, dtype)

    def get_naive_batch(self, gp, testcase, batchsize, device, dtype, **kwargs):
        new_indexs = []

        while len(new_indexs) < batchsize:
            max_index = self.get_first_max_index(gp, testcase, device, dtype)
            if max_index not in new_indexs:
                new_indexs.append(max_index)
                self.pop(max_index)

            else:
                self.pop(max_index)
                max_index = self.get_first_max_index(gp, testcase, device, dtype)

        return new_indexs

    def get_kb_batch(self, gp, testcase, batchsize, device, dtype, **kwargs):
        X_train = gp.train_inputs[0].to(device, dtype)
        new_indexs = []
        fake_x_list = torch.Tensor([]).to(device, dtype)
        fake_y_list = torch.Tensor([]).to(device, dtype)

        likelihood = kwargs["likelihood"]
        algorithmopts = kwargs["algorithmopts"]
        excursion_estimator = kwargs["excursion_estimator"]
        gp_fake = deepcopy(gp)

        while len(new_indexs) < batchsize:
            max_index = self.get_first_max_index(gp, testcase, device, dtype)

            if max_index not in new_indexs:
                new_indexs.append(max_index)
                self.pop(max_index)
                fake_x = testcase.X.to(device, dtype)[max_index].reshape(1, -1)
                fake_x_list = torch.cat((fake_x_list, fake_x), 0)

                gp_fake.eval()
                likelihood.eval()
                fake_y = likelihood(gp_fake(fake_x)).mean
                fake_y_list = torch.cat((fake_y_list, fake_y), 0)

                # print('******* train_targets', gp_fake.train_targets.dim(), gp_fake.train_targets)
                # print('******* model_batch_sample ', len(gp_fake.train_inputs[0].shape[:-2]))

                gp_fake = gp_fake.get_fantasy_model(
                    fake_x_list, fake_y_list, noise=likelihood.noise
                )

                # gp_fake = self.update_fake_posterior(
                #    testcase,
                #    algorithmopts,
                #    gp_fake,
                #    likelihood,
                #    fake_x_list,
                #    fake_y_list,
                # )

                new_acq_values = excursion_estimator.get_acq_values(gp_fake, testcase)
                self.update(new_acq_values, device, dtype)

            else:
                self.pop(max_index)
                max_index = self.get_first_max_index(gp_fake, testcase, device, dtype)

        return new_indexs

    def update_fake_posterior(
        self,
        testcase,
        algorithmopts,
        model_fake,
        likelihood,
        list_fake_xs,
        list_fake_ys,
    ):
        with torch.autograd.set_detect_anomaly(True):

            if self._n_dims == 1:
                # calculate new fake training data
                inputs = torch.cat(
                    (model_fake.train_inputs[0], list_fake_xs), 0
                ).flatten()
                targets = torch.cat(
                    (model_fake.train_targets.flatten(), list_fake_ys.flatten()), dim=0
                ).flatten()

            else:
                inputs = torch.cat((model_fake.train_inputs[0], list_xs), 0)
                targets = torch.cat(
                    (model_fake.train_targets, list_fake_ys), 0
                ).flatten()

            model_fake.set_train_data(inputs=inputs, targets=targets, strict=False)
            model_fake = get_gp(
                inputs, targets, likelihood, algorithmopts, testcase, self.device
            )

            likelihood.train()
            model_fake.train()
            fit_hyperparams(model_fake, likelihood)

        return model_fake

    def euclidean_distance_idxs(self, array_idxs, point_idx, testcase):
        array = testcase.X[array_idxs]  
        point = testcase.X[point_idx] #vector
        d = array - point
        d = torch.sqrt(torch.sum(d**2)) #vector
        d = torch.min(d).item() #USE DIST
        if(array_idxs == []):
            return 1e8
        else:
            return d #returns a scalar

    def get_distanced_batch(self, gp, testcase, batchsize, device, dtype, **kwargs):
        new_indexs = []
        #c times the minimum grid step of separation between selected points in batch
        c = 75 #has to be > 1
        step = min((testcase.rangedef[:,1] - testcase.rangedef[:,0])/testcase.rangedef[:,-1])
        distance = c * step

        while len(new_indexs) < batchsize:
            max_index = self.get_first_max_index(gp, testcase, device, dtype)
            if max_index not in new_indexs:
                if self.euclidean_distance_idxs(new_indexs, max_index, testcase)  >= distance:
                    new_indexs.append(max_index)
                    self.pop(max_index)

            else:
                self.pop(max_index)
                max_index = self.get_first_max_index(gp, testcase, device, dtype)

        return new_indexs

    

