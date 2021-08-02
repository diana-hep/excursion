from excursion_new.excursion import ExcursionProblem
from excursion_new.models import ExcursionModel


class _Estimator(object):

    def __init__(self,  problem_details: ExcursionProblem, n_funcs: int, device_type,
                 base_estimator: str or list or ExcursionModel = "TorchGP", n_initial_points=None, initial_point_generator = "random",
                 acq_func = "MES", acq_optimizer = None, acq_func_kwargs={}, acq_optimzer_kwargs={}, ):
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
