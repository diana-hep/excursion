from ._optimizer import _Optimizer
from .builders import *
import numpy as np

class Optimizer(_Optimizer):
    """Run bayesian optimisation loop.

       An `Estimator` represents the steps of a excursion set optimisation loop. To
       use it you need to provide your own loop mechanism.

       Use this class directly if you want to control the iterations of your
       bayesian optimisation loop.

       Parameters
       ----------
       base_model : str, `"GridGP"`, `"GPyTorchGP"`, or excursion custom model, \
               default: `"GPyTorchGP"`
               ## (future, list of str or ExcursionGP - better multioutput gpytorch model) ##
           Should inherit from :obj:`excursion.models.ExcursionModel`.
           It has parameters:
           train_X : Tensor, default: None
           train_y : Tensor, default: None
           likelihood: gpytorch, default: FixedNoiseGuassianLikelihood
           prior: actually a mean, will address later
           likelihood_kwargs: dict, since this will be another object
           prior_kwargs: dict, these will be the options for the mean_module

           This will allow us to have anytype of regression GP we want to make? maybe.

       base_model_kwargs : dict, default: {'likelihood': 'GaussianLikelihood', 'noise': 0.0}
            Additional arguments to be passed to the model builder.
           The type of gpytorch likelihood that should be used for the base_model.

       n_initial_points : int, default: 2
           Number of evaluations of `true func` with initialization points
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
      search_space : dict
           Dict of search space parameters.
           The search dimension is an int, and the meshgrid
           is a numpy.meshgrid or torch.meshgrid (or ndarray) of the true function
           domain that we wish to search. The thresholds is a ndarray list or Tensor.

       model : ExcursionModel (list or multioutput gp in future)
           Gaussian Process used to predict observations and compute acquisition
           function.

       """

    #
    # Some helpers for the initializer. Handles error checking
    #
    def _check_and_set_device_dtype(self):
        allowed_devices = ['skcpu']
        device = self.device_dtype['device']
        if not isinstance(device, str):
            raise TypeError("Expected device is type str, got %s" % type(device))
        elif device.lower() not in allowed_devices:
            raise ValueError("expected device_type to be in %s, got %s" % (",".join(allowed_devices), device))
        self.device_dtype['device'] = device

        allowed_dtypes = ['float64']
        dtype = self.device_dtype['dtype']
        if not isinstance(dtype, str):
            raise TypeError("Expected dtype is type str, got %s" % type(dtype))
        elif dtype not in allowed_dtypes:
            raise ValueError("expected dtype to be in %s, got %s" % (",".join(allowed_dtypes), dtype))
        self.device_dtype['dtype'] = np.float64

    # Another helper
    def _check_and_set_init_points(self):
        if not isinstance(self.n_initial_points_, int):
            raise TypeError("Expected type int, got %s" % type(self.n_initial_points_))
        elif self.n_initial_points_ <= 0:
            raise ValueError("Expected `n_initial_points` > 0, got %d" % self.n_initial_points_)
        self._n_initial_points = self.n_initial_points_

    def _check_and_set_search_space(self, details):
        _thresholds = [-np.inf] + details.thresholds + [np.inf]
        _X_pointsgrid = details.X_pointsgrid
        if self.device_dtype['device'] != 'skcpu':
            raise NotImplementedError("Only device type 'skcpu' is supported")
        self._search_space['thresholds'] = _thresholds
        self._search_space['X_pointsgrid'] = _X_pointsgrid
        self._search_space['ndim'] = details.ndim

    def __init__(self, problem_details: ExcursionProblem, device: str, dtype: str, n_funcs: int = None,
                 base_model: str or ExcursionModel = "SKLearnGP", n_initial_points=None, initial_point_generator="random",
                 acq_func: str = "PES", fit_optimizer=None, jump_start: bool = True, log: bool = True,
                 fit_optimizer_kwargs={}, acq_func_kwargs={}, base_model_kwargs={}):

        # Currently only supports 1 func
        self.n_funcs = n_funcs if n_funcs else len(problem_details.functions)
        if self.n_funcs != 1:
            raise ValueError("Expected 'n_funcs' = 1, got %d" % self.n_funcs)


        # Create the device and dtype, currently only supports string input args.
        self.device_dtype = {'device': device, 'dtype': dtype}
        self._check_and_set_device_dtype()

        # This creates a dictionary that stores the dimension of search space (int), thresholds
        # (problem_details dtype) and the search space X_grid (as problem_details dtype). It will also assign
        # them to the right device.
        self._search_space = {}
        self._check_and_set_search_space(problem_details)

        # Will check this param is set correctly and set a private n_init_points counter
        self.n_initial_points_ = n_initial_points
        self._check_and_set_init_points()

        # Configure private initial_point_generator
        self._point_sampler = build_sampler(initial_point_generator)
        self._initial_samples = None



        # These will store the result, but if the base_model was None, then this will also be None
        self.result = None
        self.log = log

        # Store acquisition function (must be a str originally, for now, if None,
        # then base_model was None, else use _point_sampler):
        self.acq_func = acq_func

        self.fit_optimizer = fit_optimizer
        if not isinstance(base_model_kwargs, dict):
            raise TypeError("Expected type dict, got %s" % type(base_model_kwargs))

        # Store the model (might be a str)
        # If not it SHOULD be that self.base_model = self._model (builder should return same self.base_model instance)
        self.base_model = base_model
        # If it was None, then tell will return a None result and behavior will be just asking for random points
        if self.base_model is not None:
            base_model_kwargs.update(self.device_dtype)
            self._model = build_model(self.base_model, rangedef=problem_details.rangedef, **base_model_kwargs)

            # for now self.acq_func is a string, so this COULD add a string to the name of the graph of plotted result
            self.result = build_result(problem_details, self.acq_func, **self.device_dtype)
            self.acq_func = build_acquisition_func(self.acq_func)

            # If I want to add all init points in a batch
            if jump_start:
                self._n_initial_points = 0
                x = self._point_sampler.generate(self.n_initial_points_, self._search_space['X_pointsgrid'])
                y = problem_details.functions[0](x)
                self._model.update_model(x, y)
                self.fit()

                # Build the result if we want to plot the initial state.
                self.result.update_result(self._model, None, None, self._search_space['X_pointsgrid'], log=self.log)
            else:
                self._initial_samples = self._point_sampler.generate(self.n_initial_points_,
                                                                     self._search_space['X_pointsgrid'])

    def ask(self, n_points=None, batch_kwarg={}):
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
            - If set to `"batch"`, then will use solve batched acquisition in ask
        """
        if n_points is None:
            return self._ask()

        if not (isinstance(n_points, int) and n_points > 0):
            raise ValueError("Expected 'n_points' is an int > 0, got %s with type %s" %
                             (str(n_points), str(type(n_points))))

        return []

    def _ask(self):
        """Suggest next point at which to evaluate the objective.
        Return a random point while not at least `n_initial_points`
        observations have been `tell`ed, after that `base_model` is used
        to determine the next point.
        """

        # after being "told" n_initial_points we switch from sampling
        # random points to using a surrogate model in update_next()
        if self._n_initial_points > 0 or self.base_model is None:
            if self._initial_samples is None:
                # Not sure I can ever get to this piece of code
                return self._point_sampler.generate(1, self._search_space['X_pointsgrid'])
            else:
                # The samples are evaluated starting form initial_samples[0]
                return self._initial_samples[
                    len(self._initial_samples) - self._n_initial_points].reshape(1, self._search_space['ndim'])

        else:

            if not hasattr(self, '_next_x'):
                self.update_next()  # Should only happen on the first call, after which _next_x should always be set.

            # Currently being found in acquisition, however, should use _next_xs and find _next_x here
            # in later refactor.
            next_x = self._next_x


            # return point computed from last call to tell()
            return next_x

    def tell(self, x, y, fit=True) -> ExcursionResult:
        """Record an observation (or several) of the objective function.
        Provide values of the objective function at points suggested by
        `ask()` or other points. By default a new model will be fit to all
        observations. The new model is used to ask the next point at
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

        # ADD THIS IN EVENTUALLY FOR SPACES WITH INVALID REGIONS
        # check_x_in_space(x, self.space)
        # self._check_y_is_valid(x, y)

        if self.base_model is not None:
            self._model.update_model(x, y)

            # after being "told" n_initial_points we switch from sampling
            # random points to using a surrogate model in update_next()
            if self._n_initial_points > 0: self._n_initial_points -= 1

            acq_vals = self.acq_func.acq_vals

            if fit:
                self.fit()
                self.update_next()  # acq happens in update_next(), it updates _next_x
            # Build result of current state, _tell will update to state n+1

            self.result.update_result(self._model, x, acq_vals, self._search_space['X_pointsgrid'], log=self.log)

        return self.result

    def fit(self):
        """Calls the model's fit method. Gives user access to this command as well.
        """
        if self.base_model is not None:
            self._model.fit_model(self.fit_optimizer)

    def update_next(self):
        """Updates the value returned by opt.ask(). Does so by calling the acquisition func. Useful if a parameter
        was updated after ask was called."""
        # self.cache_ = {}
        # Ask for a new next_x.

        if self._n_initial_points <= 0:
            self.next_xs_ = []
            next_x = self.acq_func.acquire(self._model, self._search_space['thresholds'],
                                           self._search_space['X_pointsgrid'])
            self.next_xs_.append(next_x)
            # # Placeholder until I do batch acq functions
            self._next_x = self.next_xs_[0].reshape(1, self._search_space['ndim'])

    def get_result(self):
        """Returns the most recent result as a new object. self.result stores the log if log = true.
        """
        return self.result.get_last_result()
