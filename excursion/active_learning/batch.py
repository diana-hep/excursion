import torch
import gpytorch
import os
import itertools
from excursion import get_gp, fit_hyperparams


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
        self.grid[index] = torch.Tensor([(-1.)*float("Inf")])
        #self.grid[index] = torch.empty((1), dtype=self.dtype)

    def update(self, acq_values_of_grid, device, dtype):
        self.grid = torch.as_tensor(acq_values_of_grid, device=device, dtype=dtype)

    def get_first_max_index(self, gp, testcase, device, dtype):
        X_train = gp.train_inputs[0].to(device, dtype)

        new_index = torch.argmax(self.grid)
        new_x = testcase.X.to(device, dtype)[new_index]

        # if the index is not already picked nor in the training set
        # accept it ans remove from future picks
        if (new_index not in self.picked_indexs) and (new_x.tolist() not in X_train.tolist()):
            self.pop(new_index)
            self.picked_indexs.append(new_index.item())
            return new_index.item()
        
        else:
            self.pop(new_index)
            return self.get_first_max_index(gp, testcase, device, dtype)      
        
            # is X_grid[new_index] already picked?
            #mask = torch.abs(
            #    X_train - testcase.X.to(device, dtype)[new_index]
            #)  
            #identical_elements = mask[mask.sum(dim=1) == 0]

            #number_identical_elements = identical_elements.size()[0]

            #if number_identical_elements == 0 and new_index not in self._picked_indexs :
            #    # no, accept it, stop
            #    return new_index.item()
            #    self._picked_indexs.append(new_index)
            #    continue_ = False
            #    break

            #else:
            #    count += 1
            #    continue_ = True


    def get_naive_batch(self, gp, testcase, batchsize, device, dtype, **kwargs):
        new_indexs = []

        while len(new_indexs) < batchsize:
            os.system("echo get_naive_batch")
            max_index = self.get_first_max_index(gp, testcase, device, dtype)
            if max_index not in new_indexs:
                new_indexs.append(max_index)
                #self.pop(max_index)
                continue
            #else:
                #self.pop(max_index)
            #    max_index = self.get_first_max_index(gp, testcase, device, dtype)

        return new_indexs

    def get_kb_batch(self, gp, testcase, batchsize, device, dtype, **kwargs):
        X_train = gp.train_inputs[0].to(device, dtype)
        new_indexs = []
        fake_x_list = torch.Tensor([]).to(device, dtype)
        fake_y_list = torch.Tensor([]).to(device, dtype)

        likelihood = kwargs["likelihood"]
        algorithmopts = kwargs["algorithmopts"]
        excursion_estimator = kwargs["excursion_estimator"]
        gp_fake = gp

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

                gp_fake = self.update_fake_posterior(
                    testcase,
                    algorithmopts,
                    gp_fake,
                    likelihood,
                    fake_x_list,
                    fake_y_list,
                )
                new_acq_values = excursion_estimator.get_acq_values(gp_fake, testcase)
                self.update(new_acq_values, device, dtype)

            else:
                self.pop(max_index)
                max_index = self.get_first_max_index(gp_fake, testcase, device, dtype)

        return new_indexs

    def update_fake_posterior(
        self, testcase, algorithmopts, model_fake, likelihood, list_xs, list_fake_ys
    ):

        if self._n_dims == 1:
            inputs = torch.cat((model_fake.train_inputs[0], list_xs), 0).flatten()
            targets = torch.cat((model_fake.train_targets, list_fake_ys), 0).flatten()

        else:
            inputs = torch.cat((model_fake.train_inputs[0], list_xs), 0)
            targets = torch.cat((model_fake.train_targets, list_fake_ys), 0).flatten()

        model_fake.set_train_data(inputs=inputs, targets=targets, strict=False)
        model_fake = get_gp(
            inputs, targets, likelihood, algorithmopts, testcase, self.device
        )

        likelihood.train()
        model_fake.train()
        fit_hyperparams(model_fake, likelihood)

        return model_fake


# class batchGrid(torch.Tensor):
#     def __init__(self, , device):
#         torch.Tensor.__init__(self, acq_values_of_grid, device=device)


#     def pop(self, index):
#         self[index] = -1E10


#     def get_first_max_index(self, gp, testcase, device, dtype):
#         X_train = gp.train_inputs[0].to(device, dtype)
#         continue_ = True

#         while continue_:  # until accptance of index
#             new_index = torch.argmax(self.grid)  # index with max ac value
#             mask = torch.abs(
#                 X_train - testcase.X.to(device, dtype)[new_index]
#             )  # is X_grid[new_index] already picked?
#             identical_elements = mask[mask.sum(dim=1) == 0]
#             number_identical_elements = identical_elements.size()[0]

#             print("new_index ", new_index)
#             print("x ", testcase.X[new_index])
#             print("mask ", mask)
#             print("identical_elements ", identical_elements)
#             print("number_identical_elements ", number_identical_elements)
#             print("X_train ", X_train.size(), X_train)


#             os.system("echo new_index   "+ str(new_index)) #testing
#             os.system("echo x   "+ str(testcase.X[new_index].tolist())) #testing
#             os.system("echo mask   "+ str(mask.tolist())) #testing
#             os.system("echo identical_elements   "+ str(identical_elements.tolist())) #testing
#             os.system("echo number_identical_elements   "+ str(number_identical_elements)) #testing
#             os.system("echo X_train   "+ str(X_train.tolist())) #testing


#             if number_identical_elements == 0:
#                 # no, accept it, stop
#                 return new_index.item()
#                 continue_ = False
#                 break
#             else:
#                 os.system("echo  trapped")
#                 print("trapped")
#                 # yes, discard index try again
#                 self.pop(new_index)


#     def get_naive_batch(
#         self, gp, testcase, batchsize, device, dtype, **kwargs
#     ):
#         X_train = gp.train_inputs[0].to(device, dtype)
#         new_indexs = []
#         new_ordered_indexs = batchgrid

#         while len(new_indexs) < batchsize:
#             os.system("echo get_naive_batch not reached batchsize")
#             max_index = get_first_max_index(self, gp, testcase, device, dtype)
#             if max_index not in new_indexs:
#                 os.system("echo accepted")
#                 new_indexs.append(max_index)
#                 batchgrid.pop(max_index)
#                 continue
#             else:
#                 os.system("echo rejected call get_first_max_index again")
#                 print('MAX_INDEX IN POP', max_index)
#                 self.pop(max_index)
#                 max_index = get_first_max_index(
#                     self, gp, testcase, device, dtype
#                 )

#         print("new_indexS ", new_indexs)
#         return new_indexs


#     def get_kb_batch(gp, acq_values_of_grid, testcase, batchsize, device, dtype, **kwargs):
#         X_train = gp.train_inputs[0].to(device, dtype)
#         new_indexs = []
#         new_xs = torch.Tensor([]).to(device, dtype)
#         new_fake_ys = torch.Tensor([]).to(device, dtype)
#         new_acq_values_grid = acq_values_of_grid

#         likelihood = kwargs["likelihood"]
#         algorithmopts = kwargs["algorithmopts"]
#         self = kwargs["self"]
#         gp_fake = gp

#         while len(new_indexs) < batchsize:
#             max_index = get_first_max_index(
#                 gp, new_acq_values_grid, testcase, device, dtype
#             )
#             new_indexs.append(max_index)

#             x = testcase.X.to(device, dtype)[max_index]  # .reshape(1, -1)
#             new_xs = torch.cat((new_xs, x), 0)
#             gp_fake.eval()
#             likelihood.eval()
#             y_fake = likelihood(gp_fake(x)).mean
#             new_fake_ys = torch.cat((new_fake_ys, y_fake), 0)

#             gp_fake = self.update_fake_posterior(
#                 testcase, algorithmopts, gp_fake, likelihood, new_xs, new_fake_ys
#             )
#             new_acq_values_grid = self.get_acq_values(gp_fake, testcase)

#         return new_indexs


#     def get_cluster_batch(gp, ordered_indexs, testcase, batchsize, self, **kwargs):

#         X_train = gp.train_inputs[0]
#         new_indexs = []

#         new_ordered_indexs = ordered_indexs
#         algorithmopts = kwargs["algorithmopts"]

#         while len(new_indexs) < batchsize:
#             pass

#         return new_indexs
