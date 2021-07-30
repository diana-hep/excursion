import copy
import inspect
import time, simplejson, gc
from .builders import build_sampler, build_model, build_acquisition_func
import numpy as np
import torch
from excursion_new.excursion import ExcursionProblem, ExcursionResult, build_result
from excursion_new.models import ExcursionModel
from excursion_new.models.fit import fit_hyperparams
from ._estimator import _Estimator
from collections import OrderedDict


class Optimizer(_Estimator):
    """Run bayesian optimisation loop.

       An `Estimator` represents the steps of a excursion set optimisation loop. To
       use it you need to provide your own loop mechanism.

       #The various#
       #optimisers provided by `skopt` use this class under the hood.#

       Use this class directly if you want to control the iterations of your
       bayesian optimisation loop.

       Parameters
       ----------
       dimensions : tuple, shape (n_dims, meshgrid)
           Tuple of search space dimensions.
           The search dimension is an int, and the meshgrid
           is a numpy.meshgrid or torch.meshgrid (or ndarray) of the true function
           domain that we wish to search.

       base_estimator : str, `"GridGP"`, `"ExactGP"`, or excursion custom model, \
               default: `"ExactGP"`
               ## (future, list of str or ExcursionGP - better multioutput gpytorch model) ##
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
           before approximating it with `base_model`. Initial point
           generator can be changed by setting `initial_point_generator`.

       jump_start : bool, default: False
           If True then it will add all initial points before fitting
           the model. If it false then the model will have each initial pointed
           added one by one and trained after each point.

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
           # # - `"PPES"` posterior predictive entropy search.
           # # - `"gp_hedge"` Probabilistically choose one of the above three
           # #   acquisition functions at every iteration.
           #
           #   - The gains `g_i` are initialized to zero.
           #   - At every iteration,
           #
           #     - Each acquisition function is optimised independently to
           #       propose an candidate point `X_i`.
           #     - Out of all these candidate points, the next point `X_best` is
           #       chosen by :math:`softmax(\\eta g_i)`
           #     - After fitting the surrogate model with `(X_best, y_best)`,
           #       the gains are updated such that :math:`g_i -= \\mu(X_i)`


       acq_optimizer : string, `"Adam"` or `"lbfgs"`, default: `"Adam"`
           Method to minimize the acquisition function. The fit model
           is updated with the optimal value obtained by optimizing `acq_func`
           with `acq_optimizer`.

           # - If set to `"auto"`, then `acq_optimizer` is configured on the
           #   basis of the base_model and the space searched over.
           # - If set to `"sampling"`, then `acq_func` is optimized by computing
           #   `acq_func` at `n_points` randomly sampled points.
           # - If set to `"lbfgs"`, then `acq_func` is optimized by
               -
             # - Sampling `n_restarts_optimizer` points randomly.
             # - `"lbfgs"` is run for 20 iterations with these points as initial
             #   points to find local minima.
             # - The optimal of these local minima is used to update the prior.

       random_state : int, RandomState instance, or None (default)
           Set random state to something other than None for reproducible
           results.

       n_funcs : int, default: None
           The number of true functions if provided, determines if truth values can
           be graphed or if the model can be jump started. (will have
           jump start available for provided init_y in future)

       acq_func_kwargs : dict
           Additional arguments to be passed to the acquisition function.

       acq_optimizer_kwargs : dict
           Additional arguments to be passed to the acquisition optimizer.

       Attributes
       ----------
       # Xi : list
       #     Points at which objective has been evaluated.
       # yi : scalar
       #     Values of objective at corresponding points in `Xi`.
       models : ExcursionModel (list or multioutput gp in future)
           Regression models used to fit observations and compute acquisition
           function.
       # space : Space
       #     An instance of :class:`skopt.space.Space`. Stores parameter search
       #     space used to sample points, bounds, and type of parameters.

       """

    def cook_device(self):
        if isinstance(self.device, str):
            allowed_devices = ['auto', 'cpu', 'cuda']
            self.device = self.device.lower()
            if self.device not in allowed_devices:
                raise ValueError("expected device_type to be in %s, got %s" %
                                 (",".join(allowed_devices), self.device))
            if self.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            raise TypeError("Expected type str, got %s" % type(self.device))
        self.device = torch.device(self.device)

    def cook_init_points(self):
        if self.n_initial_points_ is None:
            self.n_initial_points_ = self.details.init_n_points
        if isinstance(self.n_initial_points_, int):
            if self.n_initial_points_ < 0:
                raise ValueError(
                    "Expected `n_initial_points` > 0, got %d" % self.n_initial_points_)
        else:
            raise TypeError("Expected type int or None, got %s" % type(self.n_initial_points_))
        self._n_initial_points = self.n_initial_points_

    def __init__(self, details: ExcursionProblem, device: str, n_funcs: int = None,
                 base_estimator: str or list or ExcursionModel = "ExactGP", n_initial_points=None,
                 initial_point_generator="random", acq_func: str = "MES", acq_optimizer=None, acq_func_kwargs={},
                 acq_optimzer_kwargs={}, jump_start: bool = True):

        self.n_funcs = len(details.functions) if not n_funcs else n_funcs
        if self.n_funcs <= 0:
            raise ValueError("n_funcs must be greater than 0")

        # Create the device, currently only supports strings and initializes torch.device objects.
        self.device = device
        self.cook_device()

        # Create the special ordered dict to track iterations
        self.data_ = self._Data()

        # Will check this param is set correctly and set private n_initial_points variable
        self.n_initial_points_ = n_initial_points
        self.cook_init_points()

        # Configure initial_point_generator
        self._initial_point_generator = build_sampler(initial_point_generator)
        details.init_X_points = self._initial_samples = \
            self._initial_point_generator.generate(self._n_initial_points, details.plot_X)
        if self.device != "skcpu":
            self._initial_samples = torch.tensor(self._initial_samples, dtype=details.dtype, device=self.device)

        # Return untrained and empty model
        self.base_model = base_estimator
        self.model = build_model(self.base_model, device=self.device, dtype=details.dtype)

        # Configure acquisition function:
        details.acq_func = self.acq_func = acq_func
        if acq_func_kwargs is None:
            acq_func_kwargs = dict()
        self.epsilon = acq_func_kwargs.get("epsilon", 0.0)
        self.acq_func_kwarsgs = acq_func_kwargs

        # Some things were updated in details
        self.details = details

        # If I want to add all init points first
        if jump_start:
            self._n_initial_points = 0
            x = self._initial_samples
            y = self.details.functions[0](x)
                # self.model_acq_funcs_.append(build_acquisition_func(acq_function=self.acq_func, device=self.device, dtype=self.details.dtype))
            # Need to get a next_x so call private tell
            # Have to make sure _tell can handle lists of objects for multiple functions
            self._tell(x, y)


        # Initialize cache for `ask` method responses
        # This ensures that multiple calls to `ask` with n_points set
        # return same sets of points. Reset to {} at every call to `tell`.
        # self.cache_ = {}

    class _Data(OrderedDict):
        """ Store estimator x and y data of each model on each iteration. Tracks the order of
        insertion to be able to back step through algorithm training process. Provide speed
        advantages over list of lists (assumed, but should be O(1) access time for hashmap)
        """

        def __init__(self, maxsize=1000, /, *args, **kwds):
            self.maxsize = maxsize
            super().__init__(*args, **kwds)

        def __getitem__(self, key):
            value = super().__getitem__(key)
            self.move_to_end(key)
            return value

        def __setitem__(self, key, value):
            if key in self:
                self.move_to_end(key)
            super().__setitem__(key, value)
            if len(self) > self.maxsize:
                oldest = next(iter(self))
                del self[oldest]

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
            return self._suggest()

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
        observations have been `tell`ed, after that `base_model` is used
        to determine the next point.
        """
        if self._n_initial_points > 0 or self.base_model is None:
            # this will not make a copy of `self.rng` and hence keep advancing
            # our random state.
            if self._initial_samples is None:
                # Not sure I can ever get to this piece of code
                return self._initial_point_generator.generate(1, self.details.plot_X)
            else:
                # The samples are evaluated starting form initial_samples[0]
                self._next_x = self._initial_samples[
                    len(self._initial_samples) - self._n_initial_points].reshape(1, self.details.ndim)
                return self._next_x

        else:
            if not self.model:
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

    def tell(self, x, y, fit=True) -> ExcursionResult:
        if not isinstance(x, torch.Tensor) and self.device != "skcpu":
            x = torch.Tensor(x).to(device=self.device, dtype=self.details.dtype)
        if not isinstance(y, torch.Tensor) and self.device != "skcpu":
            y = torch.Tensor(y).to(device=self.device, dtype=self.details.dtype)
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
        # record information about computation times here, have it be stored in the
        # x data and come from acquisition

        return self._tell(x, y, fit=fit)

    def _tell(self, x, y, fit=True) -> ExcursionResult:
        """Perform the actual work of incorporating one or more new points.
        See `tell()` for the full description.
        This method exists to give access to the internals of adding points
        by side stepping all input validation and transformation."""

        # # optimizer learned something new - discard cache
        # self.cache_ = {}

        # after being "told" n_initial_points we switch from sampling
        # random points to using a surrogate model
        if fit and self._n_initial_points > 0:
            # if not self.model:
            #     self.model.update_model(x, y)
            #     self.model.fit_model(fit_hyperparams)
            #
            #         # self.model_acq_funcs_.append(build_acquisition_func(acq_function=self.acq_func, device=self.device, dtype=self.details.dtype))
            #
            # else:
            self.model.update_model(x, y)
            self.model.fit_model(fit_hyperparams)

            self._n_initial_points -= 1
            acq_test = build_acquisition_func(acq_function=self.acq_func, device=self.device, dtype=self.details.dtype)

        elif (fit and self._n_initial_points <= 0 and self.base_model is not None):
            self.next_xs_ = []
            thresholds = [-np.inf] + self.details.thresholds + [np.inf]

            ## Had to add bc memory overloaded
            acq_test = build_acquisition_func(acq_function=self.acq_func, device=self.device,
                                              dtype=self.details.dtype)

            self.model.update_model(x, y)
            self.model.fit_model(fit_hyperparams)
            next_x = acq_test.acquire(self.model, thresholds, self.details.plot_X)
            self.next_xs_.append(next_x)

            ## Placeholder until I do batch acq functions
            self._next_x = self.next_xs_[0].reshape(1, self.details.ndim)

        if isinstance(x, list):
            zipped = zip(x, y)
            if self.n_funcs > 1:
                for func in range(self.n_funcs):
                    for xi, yi in zipped:
                        self.data_[func][yi] = xi
            else:
                for xi, yi in zipped:
                    self.data_[yi] = xi
        else:
            self.data_[y] = x

        # # Pack results
        # result = create_result(self.Xi, self.yi, self.space, self.rng,
        #                       models=self.models)

        result = build_result(self.details, self.model, acq_test.log, self._next_x, device=self.device,
                              dtype=self.details.dtype)

        # result = build_result(self.details, self.models[0], self.model_acq_funcs_[0].log, self._next_x, device=self.device, dtype=self.details.dtype)

        # result.specs = self.specs
        return result

    def update_next(self):
        """Updates the value returned by opt.ask(). Useful if a parameter
        was updated after ask was called."""
        # self.cache_ = {}
        # Ask for a new next_x.
        # We only need to overwrite _next_x if it exists.
        # if hasattr(self, '_next_x'):
        #     opt = self.copy(random_state=self.rng)
        #     self._next_x = opt._next_x
