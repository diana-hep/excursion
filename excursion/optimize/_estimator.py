from excursion.excursion import ExcursionProblem
from excursion.models import ExcursionModel


class _Estimator(object):
    def __init__(self, details: ExcursionProblem, device: str, n_funcs: int = None,
                 base_estimator: str or list or ExcursionModel = "ExactGP", n_initial_points=None,
                 initial_point_generator="random", acq_func: str = "MES", fit_optimizer=None,
                 base_estimator_kwargs=None, fit_optimizer_kwargs=None, acq_func_kwargs=None, jump_start: bool = True):
        raise NotImplementedError()

    def suggest(self, n_points=None, batch_kwarg={}):
        raise NotImplementedError()

    def _suggest(self):
        raise NotImplementedError()

    def tell(self, x, y, fit=True):
        raise NotImplementedError()

    def _tell(self, x, y, fit=True):
        raise NotImplementedError()

    def update_next(self):
        raise NotImplementedError()
