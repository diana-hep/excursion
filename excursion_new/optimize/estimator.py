import copy
import inspect
import time, simplejson, gc
from .builders import build_sampler, build_model, build_acquisition_func
import numpy as np
import torch
from excursion_new.excursion import ExcursionProblem, ExcursionResult, build_result
from excursion_new.models import ExcursionModel
from ._estimator import _Estimator
from collections import OrderedDict


class Optimizer(_Estimator):
    """Run bayesian optimisation loop.

       An `Estimator` represents the steps of a excursion set optimisation loop. To
       use it you need to provide your own loop mechanism.

       Use this class directly if you want to control the iterations of your
       bayesian optimisation loop.

       Parameters
       ----------
       # dimensions : tuple, shape (n_dims, meshgrid)
       #     Tuple of search space dimensions.
       #     The search dimension is an int, and the meshgrid
       #     is a numpy.meshgrid or torch.meshgrid (or ndarray) of the true function
       #     domain that we wish to search.

       base_estimator : str, `"GridGP"`, `"TorchGP"`, or excursion custom model, \
               default: `"TorchGP"`
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

       base_estimator_kwargs : dict, default: {'likelihood': 'GaussianLikelihood', 'noise': 0.0}
            Additional arguments to be passed to the model builder.
           The type of gpytorch likelihood that should be used for a gpytorch model.
           If device type is "sk

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
           # - `"lhs"` for a latin hypercube sequence,
           # - `"grid"` for a uniform grid sequence

       acq_func : string, default: `"MES"`
           Function to minimize over the posterior distribution. Can be either

           - `"MES"` maximum entropy search
           - `"PES"` predictive entropy search
           # # - `"PPES"` posterior predictive entropy search.
           # # - `"gp_hedge"` Probabilistically choose one of the above three
           # #   acquisition functions at every iteration.

       fit_optimizer : string, `"Adam"` or `"lbfgs"`, default: `None`
           Method to train the hyperparamters of the model's Gaussian process. The fit model
           is used to compute the next acquisition function call after being fit to initial
           samples.

           Only available if ExcursionModel instance supports it. default 'None' will use
           model's default fit_optimizer.

       n_funcs : int, default: None
           The number of true functions if provided, determines if truth values can
           be graphed or if the model can be jump started. (will have
           jump start available for provided init_y in future)

       acq_func_kwargs : dict
           Additional arguments to be passed to the acquisition function.

       fit_optimizer_kwargs : dict, default: 'None'
           Additional arguments to be passed to the fit optimizer.

       Attributes
       ----------
       # Xi : list
       #     Points at which objective has been evaluated.
       # yi : scalar
       #     Values of objective at corresponding points in `Xi`.
       model : ExcursionModel (list or multioutput gp in future)
           Regression models used to fit observations and compute acquisition
           function.

       """

    #
    # Some helpers for the initializer. Handles error checking
    #
    def check_and_set_device(self):
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

    # Another helper
    def check_and_set_init_points(self):
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
                 initial_point_generator="random", acq_func: str = "MES", fit_optimizer=None,
                 base_estimator_kwargs=None, fit_optimizer_kwargs=None, acq_func_kwargs=None, jump_start: bool = True):

        # Currently only supports 1 func
        self.n_funcs = n_funcs if n_funcs else len(details.functions)
        if self.n_funcs != 1:
            raise ValueError("Expected 'n_funcs' = 1, got %d" % self.n_funcs)
            #     <= 0:
            # raise ValueError("n_funcs must be greater than 0")

        # Create the device, currently only supports strings and initializes torch.device objects.
        self.device = device
        self.check_and_set_device()
        # Create the special ordered dict to track iterations
        self.data_ = self._Data()

        # Will check this param is set correctly and set a private n_init_points counter
        self.n_initial_points_ = n_initial_points
        self.check_and_set_init_points()

        # Configure private initial_point_generator
        self._initial_point_generator = build_sampler(initial_point_generator)

        # May want to move this and only give the problem details the init x when
        # the model gets them
        details.init_X_points = self._initial_samples = \
            self._initial_point_generator.generate(self._n_initial_points, details.plot_X)
        if self.device != "skcpu":
            self._initial_samples = torch.tensor(self._initial_samples, dtype=details.dtype, device=self.device)

        # currently do not support any batches, or acq_func_kwargs. This is place holder
        # design idea is that if it is a batch, then ask will find the batch, not tell. so acq_func
        # should return acq(X) and not argmax(acq(X))
        # if acq_func_kwargs is None:
        #     acq_func_kwargs = dict()
        # # Number of points to get per call to ask()
        # self.npoints = acq_func_kwargs.get('batch_size')
        # self.acq_func_kwargs = acq_func_kwargs

        # Store acquisition function (must be a str originally):
        details.acq_func = self.acq_func = acq_func
        self.acq_func = build_acquisition_func(acq_function=self.acq_func, device=self.device, dtype=details.dtype)

        # Some things were updated in details
        self.details = details

        if base_estimator_kwargs is None:
            raise ValueError("base_estimator_kwargs cannot be type None")

        # Store the model (might be a str)
        # If not it SHOULD be that self.base_model = self.model (builder should return same self.base_model instance)
        self.base_model = base_estimator
        # If it was None, then tell will return a None result and behavior will be just asking for random points
        if self.base_model is not None:
            base_estimator_kwargs['device'] = self.device
            base_estimator_kwargs['dtype'] = self.details.dtype
            self.model = build_model(self.base_model, grid=details.plot_rangedef, **base_estimator_kwargs)

        self.fit_optimizer = fit_optimizer

        # If I want to add all init points first
        if jump_start:
            self._n_initial_points = 0
            x = self._initial_samples
            y = self.details.functions[0](x)
            # Need to get a next_x so call private tell
            # Have to make sure _tell can handle lists of objects for multiple init data
            self.tell(x, y)

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
                "Expected 'n_points' is an int > 0, got %s with type %s" % (str(n_points), str(type(n_points)))
            )
        pass

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
        # else:
        #     # Need to decide how to handle batches
        #     X_new = [self.model_acq_funcs_[idx].acquire(model, thresholds) for idx, model in enumerate(self.models)]
        #     X.append(X_new)
        #     return X
        #     # Caching the result with n_points not None. If some new parameters
        #     # are provided to the ask, the cache_ is not used.
        #     if (n_points, strategy) in self.cache_:
        #         return self.cache_[(n_points, strategy)]
        #
        #     # Copy of the optimizer is made in order to manage the
        #     # deletion of points with "lie" objective (the copy of oiptimizer is simply discarded)
        #     opt = self.copy(random_state=self.rng.randint(0,np.iinfo(np.int32).max))
        #
        #     for i in range(n_points):
        #         X_new = self.suggest()
        #         X.append(X_new)
        #         self._tell(X_new, y_lie)

        # self.cache_ = {(n_points, strategy): X}  # cache_ the result

        return []

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
                return self._initial_samples[
                    len(self._initial_samples) - self._n_initial_points].reshape(1, self.details.ndim)

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
        result = None

        if self.base_model is not None:
            self.model.update_model(x, y)
            if self._n_initial_points > 0: self._n_initial_points -= 1
            if fit: self.model.fit_model(self.fit_optimizer)
            # Build result of current state, _tell will update to state n+1
            result = build_result(self.details, self.model, self.acq_func.log, x, device=self.device,
                                  dtype=self.details.dtype)
            # after being "told" n_initial_points we switch from sampling
            # random points to using a surrogate model
            if self._n_initial_points <= 0:
                self.next_xs_ = []
                thresholds = [-np.inf] + self.details.thresholds + [np.inf]
                next_x = self.acq_func.acquire(self.model, thresholds, self.details.plot_X)
                self.next_xs_.append(next_x)
                # # Placeholder until I do batch acq functions
                self._next_x = self.next_xs_[0].reshape(1, self.details.ndim)

        if isinstance(x, list):
            zipped = zip(x, y)
            for xi, yi in zipped:
                self.data_[yi] = xi
        else:
            self.data_[y] = x

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
