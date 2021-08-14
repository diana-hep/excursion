from excursion.excursion import ExcursionProblem
from excursion.models import ExcursionModel


class _Optimizer(object):
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
