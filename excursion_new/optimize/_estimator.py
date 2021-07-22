import copy
import inspect
import time, simplejson, gc
from .builders import build_sampler, build_model_init, build_model, build_acquisition_func
import numpy as np
import torch
from excursion_new.excursion import ExcursionProblem
from excursion_new.models import ExcursionModel
from excursion_new.models.fit import fit_hyperparams



class Optimizer(object):
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

       base_estimator : str, `"GridGP"`, `"ExactGP"`, or excursion custom model, \
               default: `"ExactGP"`
               ## (future, list of str or ExcursionGP) ##
           Should inherit from :obj:`excursion.models.ExcursionGP`.
           Which should be initialized before hand
           It has parameters:
           train_X : Tensor, default: None
           train_y : Tensor, default: None
           likelihood: gpytorch, default: FixedNoiseGuassianLikelihood
           prior: actually a mean, will address later
           likelihood_kwargs: dict, since this will be another object
           prior_kwargs: dict, these will be the options for the mean_module

           This will allow us to have anytype of regression GP we want to make? maybe.

       n_initial_points : int or None, default: 2
           Number of evaluations of `func` with initialization points
           before approximating it with `base_estimator`. Initial point
           generator can be changed by setting `initial_point_generator`.

       initial_point_generator : str, SampleGenerator instance, \
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
    def __init__(self,  problem_details: ExcursionProblem, n_funcs: int, device_type,
                 base_estimator: str or list or ExcursionModel = "ExactGP", n_initial_points=None, initial_point_generator = "random",
                 acq_func = "MES", acq_optimizer = None, acq_func_kwargs={}, acq_optimzer_kwargs={}, ):
        self.specs = {'args': copy.copy(inspect.currentframe().f_locals), 'function': 'Optimizer'}

        # Configure acquisition function set:
        self.device = device_type
        self.acq_func = acq_func
        self.acq_func_kwarsgs = acq_func_kwargs
        self.n_funcs = len(problem_details.functions)
        allowed_acq_funcs = ["PES", "MES"]
        if self.acq_func not in allowed_acq_funcs:
            raise ValueError("expected acq_func to be in %s, got %s" %
                             (",".join(allowed_acq_funcs), self.acq_func))
        problem_details.acq_func = acq_func
        self.cand_acq_funcs_ = [self.acq_func]

        if acq_func_kwargs is None:
            acq_func_kwargs = dict()
        self.eta = acq_func_kwargs.get("eta", 1.0)

        if isinstance(n_initial_points, int) and n_initial_points < 0:
            raise ValueError(
                "Expected `n_initial_points` > 0, got %d" % n_initial_points)
        elif n_initial_points is None:
            n_initial_points = problem_details.init_n_points
        else:
            raise TypeError("Expected type int or None, got %s" % type(n_initial_points))
        self._n_initial_points = n_initial_points
        self.n_initial_points_ = n_initial_points

        # Configure initial_point_generator

        self._initial_samples = None
        self._initial_point_generator = initial_point_generator

        if isinstance(self._initial_point_generator, str):
            allowed_init_point_generator = ["random", "latin_grid", "latin_hypercube"]
            if self._initial_point_generator not in allowed_init_point_generator:
                raise ValueError("expected initial_point_generator to be in %s, got %s" %
                                 (",".join(allowed_acq_funcs), self.acq_func))

        self._initial_point_generator = build_sampler(self._initial_point_generator)
        problem_details.init_X_points = self._initial_point_generator.generate(self._n_initial_points, problem_details.plot_X)
        self._initial_samples = torch.Tensor(problem_details.init_X_points).to(dtype=problem_details.data_type, device=self.device)

        self.models = []
        self.model_acq_funcs_ = []
        self.base_estimator = base_estimator

        # build base_estimator if doesn't exist
        if isinstance(self.base_estimator, str):
            allowed_base_estimators = ["ExactGP", "GridGP"]
            if self.base_estimator not in base_estimator:
                raise ValueError("expected base_estimator to be in %s, got %s" %
                                 (",".join(allowed_acq_funcs), self.acq_func))
        # for idx, f in enumerate(problem_details.functions):
        #     self.models.append(build_model_init(base_estimator, problem_details.init_X_points, device=device_type, dtype=problem_details.data_type, n_init_points=n_initial_points, true_function=f))
        #     self.model_acq_funcs_.append(build_acquisition_func(acq_function="MES"))

        self.Xi = []
        self.yi = []
        self.problem_details = problem_details

        # Initialize cache for `ask` method responses
        # This ensures that multiple calls to `ask` with n_points set
        # return same sets of points. Reset to {} at every call to `tell`.
        # self.cache_ = {}


    def suggest(self, n_points=None, batch_kwarg={}):
        """Query point or multiple points at which objective should be evaluated.
        n_points : int or None, default: None
            Number of points returned by the ask method.
            If the value is None, a single point to evaluate is returned.
            Otherwise a list of points to evaluate is returned of size
            n_points. This is useful if you can evaluate your objective in
            parallel, and thus obtain more objective function evaluations per
            unit of time.
        batch_kwarg : dict, default: None
            Options for batch acquisition of multiple points (see also `n_points`
            description). This parameter is ignored if n_points = None.
            Supported options are `"kb"`, or `"naive"`.
            - If set to `"batch"`, then constant liar strategy is used
               with lie objective ????? value being minimum of observed objective ????
                   # # values. `"cl_mean"` and `"cl_max"` means mean and max of values # #
                   # # respectively. For details on this strategy see: # #
                   # # https://hal.archives-ouvertes.fr/hal-00732512/document # #
                   # # With this strategy a copy of optimizer is created, which is # #
                   # # then asked for a point, and the point is told to the copy of # #
                   # # optimizer with some fake objective (lie), the next point is # #
                   # # asked from copy, it is also told to the copy with fake # #
                   # # objective and so on. The type of lie defines different # #
                   # # flavours of `cl_x` strategies. # #
        """
        if n_points is None:
            return self._suggest().reshape(1, self.problem_details.ndim)


        if not (isinstance(n_points, int) and n_points > 0):
            raise ValueError(
                "n_points should be int > 0, got " + str(n_points)
            )

        X = []

        if 'batch_type' in batch_kwarg.keys():
            supported_batch_types = ["kb", "naive"]
            if not isinstance(batch_kwarg['batch_type'], str):
                raise TypeError("Expected batch_type to be one of type str" +
                                " got %s" % str(type(batch_kwarg['batch_type']))
                )
            if batch_kwarg['batch_type'] not in supported_batch_types:
                raise ValueError(
                    "Expected batch_type to be one of " +
                    str(supported_batch_types) + ", " + "got %s" % batch_kwarg['batch_type']
                )
        else:
            ## Need to decide how to handle batches
            # X_new = [self.model_acq_funcs_[idx].acquire(model, thresholds) for idx, model in enumerate(self.models)]
            # X.append(X_new)
            # return X
            # # Caching the result with n_points not None. If some new parameters
            # # are provided to the ask, the cache_ is not used.
            # if (n_points, strategy) in self.cache_:
            #     return self.cache_[(n_points, strategy)]

            # Copy of the optimizer is made in order to manage the
            # deletion of points with "lie" objective (the copy of
            # oiptimizer is simply discarded)
            # opt = self.copy(random_state=self.rng.randint(0,
            #                                               np.iinfo(np.int32).max))


            for i in range(n_points):
                X_new = self.suggest()
                X.append(X_new)
                # self._tell(X_new, y_lie)

        # self.cache_ = {(n_points, strategy): X}  # cache_ the result

        return X

    def _suggest(self):
        """Suggest next point at which to evaluate the objective.
        Return a random point while not at least `n_initial_points`
        observations have been `tell`ed, after that `base_estimator` is used
        to determine the next point.
        """
        if self._n_initial_points > 0 or self.base_estimator_ is None:
            # this will not make a copy of `self.rng` and hence keep advancing
            # our random state.
            if self._initial_samples is None:
                # Not sure I can ever get to this piece of code
                return self._initial_point_generator.generate(1, self.problem_details.plot_X)
            else:
                # The samples are evaluated starting form initial_samples[0]
                return self._initial_samples[
                    len(self._initial_samples) - self._n_initial_points]

        else:
            if not self.models:
                raise RuntimeError("Random evaluations exhausted and no "
                                   "model has been fit.")

            next_x = self._next_x
            # min_delta_x = min([self.space.distance(next_x, xi)
            #                    for xi in self.Xi])
            # if abs(min_delta_x) <= 1e-8:
            #     warnings.warn("The objective has been evaluated "
            #                   "at this point before.")

            # return point computed from last call to tell()
            return next_x

    def tell(self, x, y, fit=True):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x).to(device=self.device, dtype=self.problem_details.data_type)
        if not isinstance(y, torch.Tensor):
            y = torch.Tensor(y).to(device=self.device, dtype=self.problem_details.data_type)
        """Record an observation (or several) of the objective function.
        Provide values of the objective function at points suggested by
        `ask()` or other points. By default a new model will be fit to all
        observations. The new model is used to suggest the next point at
        which to evaluate the objective. This point can be retrieved by calling
        `ask()`.
        To add observations without fitting a new model set `fit` to False.
        To add multiple observations in a batch pass a list-of-lists for `x`
        and a list of scalars for `y`.
        Parameters
        ----------
        x : list or list-of-lists
            Point at which objective was evaluated.
        y : scalar or list
            Value of objective at `x`.
        fit : bool, default: True
            Fit a model to observed evaluations of the objective. A model will
            only be fitted after `n_initial_points` points have been told to
            the optimizer irrespective of the value of `fit`.
        """
        # check_x_in_space(x, self.space)
        # self._check_y_is_valid(x, y)

        # take the logarithm of the computation times
        # if "ps" in self.acq_func:
        #     if is_2Dlistlike(x):
        #         y = [[val, log(t)] for (val, t) in y]
        #     elif is_listlike(x):
        #         y = list(y)
        #         y[1] = log(y[1])

        return self._tell(x, y, fit=fit)

    def _tell(self, x, y, fit=True):
        """Perform the actual work of incorporating one or more new points.
        See `tell()` for the full description.
        This method exists to give access to the internals of adding points
        by side stepping all input validation and transformation."""


        # if "ps" in self.acq_func:
        #     if is_2Dlistlike(x):
        #         self.Xi.extend(x)
        #         self.yi.extend(y)
        #         self._n_initial_points -= len(y)
        #     elif is_listlike(x):
        #         self.Xi.append(x)
        #         self.yi.append(y)
        #         self._n_initial_points -= 1
        # # if y isn't a scalar it means we have been handed a batch of points
        # elif is_listlike(y) and is_2Dlistlike(x):
        #     self.Xi.extend(x)
        #     self.yi.extend(y)
        #     self._n_initial_points -= len(y)
        # elif is_listlike(x):
        #     self.Xi.append(x)
        #     self.yi.append(y)
        #     self._n_initial_points -= 1
        # else:
        #     raise ValueError("Type of arguments `x` (%s) and `y` (%s) "
        #                      "not compatible." % (type(x), type(y)))

        # # optimizer learned something new - discard cache
        # self.cache_ = {}

        # after being "told" n_initial_points we switch from sampling
        # random points to using a surrogate model
        if (fit and self._n_initial_points > 0):

            if not self.Xi:
                for idx, f in enumerate(self.problem_details.functions):
                    self.models.append(build_model(self.base_estimator, x, y, device=self.device, dtype=self.problem_details.data_type))
                    self.model_acq_funcs_.append(build_acquisition_func(acq_function="MES"))

            else:
                for idx, model in enumerate(self.models):
                    self.models[idx] = self.update(model, x, y)

            self._n_initial_points -= 1
            self.Xi.append(x)
            self.yi.append(y)
        elif (fit and self._n_initial_points <= 0 and
                self.base_estimator_ is not None):
            thresholds = [-np.inf] + self.problem_details.thresholds.tolist() + [np.inf]
            self.Xi.append(x)
            self.yi.append(y)

            def evaluate(self, x, y):
                pass

            self.next_xs_ = []
            zipped = zip(self.models, self.model_acq_funcs_)
            for idx, (model, model_acq_func) in enumerate(zipped):
                self.models[idx] = self.update(model, x, y)
                next_x = model_acq_func.acquire(self.models[idx], thresholds, self.problem_details.plot_X)
                self.next_xs_.append(next_x)

        ## Placeholder until I do multiple functions
            self._next_x = self.next_xs_[0]

        # # Pack results
        # result = create_result(self.Xi, self.yi, self.space, self.rng,
        #                        models=self.models)

        # result.specs = self.specs
        # return result
        return None


    def update(self, model, x, y):

        if x.shape[1] == 1:
            inputs_i = torch.cat(
                (model.train_inputs[0], x), dim=0).flatten()
            targets_i = torch.cat(
                (model.train_targets.flatten(), y.flatten()), dim=0).flatten()
        else:
            inputs_i = torch.cat(
                (model.train_inputs[0], x), 0)
            targets_i = torch.cat(
                (model.train_targets, y), 0).flatten()

        likelihood = model.likelihood
        model.set_train_data(inputs=inputs_i, targets=targets_i, strict=False)
        model.train()
        likelihood.train()
        fit_hyperparams(model)


        return model