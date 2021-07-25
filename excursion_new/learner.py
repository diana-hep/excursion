import optimize.estimator as est
from .excursion import ExcursionProblem


class _Learner(object):
    def __init__(self, details: ExcursionProblem = None, algorithm_options: dict = None):
        self.details = details
        self.options = algorithm_options

    def initialize(self, snapshot=False):
        if snapshot:
            raise NotImplementedError("Must initialize estimator with stored data in details")
        else:
            raise NotImplementedError("Must initialize estimator with initial parameters in details")

    def suggest(self, npoints: int = None):
        """
        Suggest a new point to evaluate.
        Parameters
        ----------
        npoints : int, default: None
            Determine how many points to suggest for batch acquisition.
        Returns
        -------
        points : object
            Some kind of array object e.g. torch.Tensor or numpy.ndarray
        """
        raise NotImplementedError()

    def tell(self, x, y):
        raise NotImplementedError()

    def evaluate(self, x):
        raise NotImplementedError()

    def evaluate_and_tell(self, x):
        raise NotImplementedError()

    def evaluate_metrics(self):
        raise NotImplementedError()


class Learner(_Learner):
    def __init__(self, testcase):
        super(Learner, self).__init__(details=ExcursionProblem(testcase.true_functions, ndim=testcase.ndim,
                                                               thresholds=testcase.thresholds,
                                                               bounding_box=testcase.bounding_box,
                                                               plot_npoints=testcase.plot_npoints))

    def initialize(self, snapshot=False):
        if snapshot:
            raise NotImplementedError("Must initialize estimator with stored data in details")
        else:
            self.estimator = est.Optimizer(problem_details=self.details, n_funcs=1, device_type="cuda")

    def suggest(self, npoints: int = None):
        """
        Suggest a new point to evaluate.
        Parameters
        ----------
        npoints : int, default: None
            Determine how many points to suggest for batch acquisition.
        Returns
        -------
        points : object
            Some kind of array object e.g. torch.Tensor or numpy.ndarray
        """
        return self.estimator.suggest()

    def tell(self, x, y):
        return self.estimator.tell(x, y)

    def evaluate(self, x):
        return self.details.functions[0](x)

    def evaluate_and_tell(self, x):
        y = self.evaluate(x)
        return self.tell(x, y)

    def evaluate_metrics(self):
        raise NotImplementedError()