import copy
import inspect
import time, simplejson, gc
import numpy as np
import torch
from sklearn.utils import check_random_state
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from sklearn.metrics import confusion_matrix
import excursion.models.gp
# from .plotting import *
from excursion.models.gp import get_gp
from excursion.acquisition import *
from excursion.models.fit import fit_hyperparams


class _Estimator:
    """Run bayesian optimisation loop.

       An `Estimator` represents the steps of a excursion set optimisation loop. To
       use it you need to provide your own loop mechanism.

       #The various#
       #optimisers provided by `skopt` use this class under the hood.#

       Use this class directly if you want to control the iterations of your
       bayesian optimisation loop.

       Parameters
       ----------
       dimensions : list, shape (n_dims,)
           List of search space dimensions.
           Each search dimension can be defined either as

           - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
             dimensions),
           - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
             dimensions),

           ##                                                              ##
           ## - as a list of categories (for `Categorical` dimensions), or ##
           ## - an instance of a `Dimension` object (`Real`, `Integer` or  ##
           ##  `Categorical`).                                             ##
           ##                                                              ##

       models : `"GP"`, `"RF"`, `"ET"`, `"GBRT"` or sklearn regressor, \
               default: `"GP"`
           Should inherit from :obj:`excursion.models.ExcursionGP`.
           Which should be initialized before hand
           It has parameters:
           train_X : Tensor, default: None
           train_y : Tensor, default: None
           likelihood: gpytorch, default: FixedNoiseGuassianLikelihood
           prior: actually a mean, will address later
           likelihood_kwargs: dict, since this will be another object
           prior_kwargs: dict, these will be the options for the mean_mowdule

           This will allow us to have anytype of regression GP we want to make? maybe.

       n_initial_points : int, default: 10
           Number of evaluations of `func` with initialization points
           before approximating it with `base_estimator`. Initial point
           generator can be changed by setting `initial_point_generator`.

       initial_point_generator : str, InitialPointGenerator instance, \
               default: `"random"`
           Sets a initial points generator. Can be either

           - `"random"` for uniform random numbers,
           # # - `"sobol"` for a Sobol sequence,
           # # - `"halton"` for a Halton sequence,
           # # - `"hammersly"` for a Hammersly sequence,
           - `"lhs"` for a latin hypercube sequence,
           - `"grid"` for a uniform grid sequence

       acq_func : string, default: `"MES"`
           Function to minimize over the posterior distribution. Can be either

           - `"MES"` maximum entropy search
           - `"PES"` predictive entropy search
           - `"PPES"` posterior predictive entropy search.
           - `"gp_hedge"` Probabilistically choose one of the above three
             acquisition functions at every iteration.

             - The gains `g_i` are initialized to zero.
             - At every iteration,

               - Each acquisition function is optimised independently to
                 propose an candidate point `X_i`.
               - Out of all these candidate points, the next point `X_best` is
                 chosen by :math:`softmax(\\eta g_i)`
               - After fitting the surrogate model with `(X_best, y_best)`,
                 the gains are updated such that :math:`g_i -= \\mu(X_i)`


       acq_optimizer : string, `"Adam"` or `"lbfgs"`, default: `"Adam"`
           Method to minimize the acquisition function. The fit model
           is updated with the optimal value obtained by optimizing `acq_func`
           with `acq_optimizer`.

           - If set to `"auto"`, then `acq_optimizer` is configured on the
             basis of the base_estimator and the space searched over.
             If the space is Categorical or if the estimator provided based on
             tree-models then this is set to be `"sampling"`.
           - If set to `"sampling"`, then `acq_func` is optimized by computing
             `acq_func` at `n_points` randomly sampled points.
           - If set to `"lbfgs"`, then `acq_func` is optimized by
               -
             # - Sampling `n_restarts_optimizer` points randomly.
             # - `"lbfgs"` is run for 20 iterations with these points as initial
             #   points to find local minima.
             # - The optimal of these local minima is used to update the prior.

       random_state : int, RandomState instance, or None (default)
           Set random state to something other than None for reproducible
           results.

       n_funcs : list, default: None
           The number of functions to run in parallel in the base_estimator,
           if the base_estimator supports n_jobs as parameter and
           base_estimator was given as string.
           If -1, then the number of jobs is set to the number of cores.

       acq_func_kwargs : dict
           Additional arguments to be passed to the acquisition function.

       acq_optimizer_kwargs : dict
           Additional arguments to be passed to the acquisition optimizer.

       Attributes
       ----------
       Xi : list
           Points at which objective has been evaluated.
       yi : scalar
           Values of objective at corresponding points in `Xi`.
       models : list
           Regression models used to fit observations and compute acquisition
           function.
       space : Space
           An instance of :class:`skopt.space.Space`. Stores parameter search
           space used to sample points, bounds, and type of parameters.

       """
    def __init__(self, dimensions, models: list, n_initial_points: int, initial_point_generator: str,
                 acq_func: str, acq_optimizer: str, random_state: int, n_funcs: list,
                 acq_func_kwargs: dict, acq_optimzer_kwargs: dict):
        self.specs = {'args': copy.copy(inspect.currentframe().f_locals), 'function': 'Estimator'}
        self.rng = check_random_state(random_state)

        # Configure acquisition function set:
        self.acq_func = acq_func
        self.acq_func_kwarsgs = acq_func_kwargs

        allowed_aqc_functions = ["gp_hedge", "PPES", "PES", "MES"]
        if self.acq_func not in allowed_aqc_functions:
            raise ValueError("ex                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               ")




    def test(self, testcase, algorithmopts, model, likelihood, device):
        self.x_new = torch.zeros(1, testcase.n_dims, dtype=torch.float64)
        self.y_new = torch.zeros(1, 1, dtype=torch.float64)
        self.acq_values = []

        self.this_iteration = 0
        self.confusion_matrix = []
        self.pct_correct = []
        self.walltime_step = []
        self.walltime_posterior = []
        self.device = device
        self.dtype = torch.float64

        self._acq_type = algorithmopts["acq"]["acq_type"]
        self._X_grid = testcase.X.to(self.device, self.dtype)
        self._epsilon = algorithmopts["likelihood"]["epsilon"]
        self._n_dims = testcase.n_dims

    def get_diagnostics(self, testcase, model, likelihood):
        thresholds = [-np.inf] + testcase.thresholds.tolist() + [np.inf]
        X_eval = testcase.X

        # noise_dist = MultivariateNormal(
        #    torch.zeros(len(X_eval)), torch.eye(len(X_eval))
        # )

        # noise = self._epsilon * noise_dist.sample(torch.Size([])).to(
        #    self.device, self.dtype
        # )

        noise = self._epsilon * Normal(
            torch.tensor([0.0]), torch.tensor([1.0])
        ).rsample(sample_shape=torch.Size([len(X_eval)]))

        # y_true = testcase.true_functions[0](X_eval).to(
        #    self.device, self.dtype
        # ) + noise.to(self.device, self.dtype)

        y_true = testcase.true_functions[0](X_eval).to(self.device, self.dtype)

        model.eval()
        likelihood.eval()
        y_pred = likelihood(model(X_eval.to(self.device, self.dtype))).mean.to(
            self.device, self.dtype
        )

        def label(y):
            for j in range(len(thresholds) - 1):
                if y < thresholds[j + 1] and y >= thresholds[j]:
                    return int(j)

        labels_pred = np.array([label(y) for y in y_pred])
        isnan_vector = np.isnan(labels_pred)

        labels_true = np.array([label(y) for y in y_true])

        # for y in labels_true:
        #    print('label ', y, type(y))

        # force y_true = y_train for those x in dataset

        conf_matrix = confusion_matrix(labels_true, labels_pred)
        self.confusion_matrix.append(conf_matrix)
        pct = np.diag(conf_matrix).sum() * 1.0 / len(X_eval)
        self.pct_correct.append(pct)
        print("pct ", pct)
        return None

    def step(self, testcase, algorithmopts, model, likelihood):
        # track wall time
        start_time = time.process_time()
        self.this_iteration += 1

        # print("Iteration ", self.this_iteration)

        ################################## this should be all one step with output
        ################################## number of batches, ordered max indices in grid

        acq_values_of_grid = self.get_acq_values(model, testcase)
        # print('ACQ VALUES')
        # print(acq_values_of_grid)

        batchgrid = batchGrid(
            acq_values_of_grid,
            device=self.device,
            dtype=self.dtype,
            n_dims=self._n_dims,
        )
        batchgrid.update(acq_values_of_grid, self.device, self.dtype)

        if algorithmopts["acq"]["batch"]:
            batchsize = algorithmopts["acq"]["batchsize"]
            batchtype = algorithmopts["acq"]["batchtype"]
            new_indexs = batchgrid.batch_types[batchtype](
                model,
                testcase,
                batchsize,
                self.device,
                self.dtype,
                likelihood=likelihood,
                algorithmopts=algorithmopts,
                excursion_estimator=self,
            )
            self.x_new = (
                torch.stack([testcase.X[index] for index in new_indexs])
                    .to(self.device, self.dtype)
                    .reshape(batchsize, self._n_dims)
            )

            # .reshape(batchsize, self._n_dims)

            # self.x_new = (testcase.X[new_indexs]).reshape(batchsize, self._n_dims).to(self.device, self.dtype)

        else:
            new_index = batchgrid.get_first_max_index(
                model, testcase, self.device, self.dtype
            )
            self.x_new = (
                testcase.X[new_index]
                    .reshape(1, self._n_dims)
                    .to(self.device, self.dtype)
            )

        ##################################

        gc.collect()
        torch.cuda.empty_cache()

        # get y from selected x
        # This is our expensive call to a black_box function

        noise_dist = MultivariateNormal(torch.zeros(1), torch.eye(1))
        noise = self._epsilon * noise_dist.sample(torch.Size([])).to(self.device, self.dtype)
        self.y_new = (testcase.true_functions[0](self.x_new).to(self.device, self.dtype)
                      + noise)

        # track wall time
        end_time = time.process_time() - start_time
        self.walltime_step.append(end_time)

        # print("x_new ", self.x_new.size(), self.x_new)
        # print("y_new ", self.y_new.size(), self.y_new)

        return self.x_new, self.y_new

    def get_acq_values(self, model, testcase):

        thresholds = [-np.inf] + testcase.thresholds.tolist() + [np.inf]

        start_time = time.time()

        acquisition_values_grid = acquisition_functions[self._acq_type](
            model, testcase, thresholds, self.device, self.dtype)

        end_time = time.time() - start_time

        # Used for plotting. must plot after this call to step
        self.acq_values = acquisition_values_grid
        # print(f"the MES acquistion_values_grid is a: {type(acquisition_values_grid)}")
        # print(f"the MES \"          \" size is: {acquisition_values_grid.size()}")
        # print(f"the MES \"          \" is: {acquisition_values_grid}")
        #

        return acquisition_values_grid

    def update_posterior(self, testcase, algorithmopts, model, likelihood):
        # track wall time
        start_time = time.process_time()
        if self._n_dims == 1:
            inputs_i = torch.cat(
                (model.train_inputs[0], self.x_new), dim=0).flatten()
            targets_i = torch.cat(
                (model.train_targets.flatten(), self.y_new.flatten()), dim=0).flatten()
        else:
            inputs_i = torch.cat(
                (model.train_inputs[0], self.x_new), 0)
            targets_i = torch.cat(
                (model.train_targets, self.y_new), 0).flatten()

        #
        #
        # What is this line doing here?
        # model.set_train_data(inputs=inputs_i, targets=targets_i, strict=False)
        #
        #
        # model = get_gp(
        #    inputs_i, targets_i, likelihood, algorithmopts, testcase, self.device)

        model.set_train_data(inputs=inputs_i, targets=targets_i, strict=False)
        model.train()
        likelihood.train()
        fit_hyperparams(model, likelihood)

        # track wall time
        end_time = time.process_time() - start_time
        self.walltime_posterior.append(end_time)

        return model
