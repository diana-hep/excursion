from .optimizer import Optimizer
from .excursion import ExcursionProblem
from .plotting import plot


class _Learner(object):
    def __init__(self, problem_details: ExcursionProblem = None, algorithm_options: dict = None):
        self.problem_details = problem_details
        self.options = algorithm_options

    def ask(self, npoints: int = None):
        """
        Suggest a new point to evaluate.
        Parameters
        ----------
        npoints : int, default: None
            Determine how many points to ask for batch acquisition.
        Returns
        -------
        points : object
            Some kind of array object e.g. torch.Tensor or numpy.ndarray
        """
        raise NotImplementedError()

    def tell(self, x, y, fit=True):
        raise NotImplementedError()

    def evaluate(self, x):
        raise NotImplementedError()

    def evaluate_and_tell(self, x, fit=True):
        raise NotImplementedError()

    def run(self, n_iterations, plot_result=False):
        raise NotImplementedError()

    def evaluate_metrics(self):
        raise NotImplementedError()


class Learner(_Learner):
    def __init__(self, problem_details, options):
        super(Learner, self).__init__(problem_details=problem_details, algorithm_options=options)
        self.optimizer = Optimizer(problem_details=self.problem_details, base_model=self.options['model']['type'],
                                       acq_func=self.options['acq']['acq_type'],
                                       jump_start=self.options['jump_start'], device=self.options['device'],
                                       n_initial_points=self.options['ninit'],
                                       initial_point_generator=self.options['init_type'],
                                       fit_optimizer=self.options['model']['fit_optimizer'],
                                       base_model_kwargs=self.options['likelihood'], dtype=self.options['dtype'])

    def ask(self, npoints: int = None):
        """
        Suggest a new point to evaluate.
        Parameters
        ----------
        npoints : int, default: None
            Determine how many points to ask for batch acquisition.
        Returns
        -------
        points : object
            Some kind of array object e.g. torch.Tensor or numpy.ndarray
        """
        return self.optimizer.ask()

    def tell(self, x, y, fit=True):
        return self.optimizer.tell(x, y, fit=fit)

    def evaluate(self, x):
        return self.problem_details.functions[0](x)

    def evaluate_and_tell(self, x, fit=True):
        y = self.evaluate(x)
        return self.tell(x, y, fit=fit)

    def run(self, n_iterations, plot_result=False, show_confusion_matrix=False):
        for iter in range(n_iterations):
            result = self.evaluate_and_tell(self.ask())
            if plot_result: plot(result, show_confusion_matrix)

    def evaluate_metrics(self):
        raise NotImplementedError()
