from excursion.excursion import ExcursionProblem
from excursion.models import ExcursionModel


class _Optimizer(object):
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
    def __init__(self, problem_details: ExcursionProblem, device: str, n_funcs: int = None,
                 base_model: str or ExcursionModel = "ExactGP", n_initial_points=None, initial_point_generator="random",
                 acq_func: str = "MES", fit_optimizer=None, jump_start: bool = True, log: bool = True,
                 fit_optimizer_kwargs=None, acq_func_kwargs=None, base_model_kwargs=None):

        raise NotImplementedError()

    def ask(self, n_points=None, batch_kwarg={}):
        raise NotImplementedError()

    def _ask(self):
        raise NotImplementedError()

    def tell(self, x, y, fit=True):
        raise NotImplementedError()

    def _tell(self, x, y, fit=True):
        """Perform the actual work of incorporating one or more new points.
        See `tell()` for the full description.
        This method exists to give access to the internals of adding points
        by side stepping all input validation and transformation."""
        raise NotImplementedError()

    def update_next(self):
        """Updates the value returned by opt.ask(). Does so by calling the acquisition func. Useful if a parameter
        was updated after ask was called."""
        raise NotImplementedError()

    def fit(self):
        """Calls the model's fit method. Give user access to this command as well.
        """
        raise NotImplementedError()

    def get_result(self):
        """Returns the most recent result as a new object. self.result stores the log if log = true.
        """
        raise NotImplementedError()
