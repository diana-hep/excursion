from collections import defaultdict


class ExcursionModel(object):
    """
    All 'excursion' models used by the library should implement this abstract interface in order to be compatible
    with the Optimizer class.
    """
    def fit_model(self, fit_optimizer):
        raise NotImplementedError()

    def update_model(self, x, y):
        raise NotImplementedError()

    def set_params(self, **params):
        """
        Set the parameters of this ExcursionModel.

        Args:
            **params (:obj:'dict'): ExcursionModel parameters.

        .. note::

            ''params'' is  a dictionary containing model specific parameters that must be set when initializing a new
            instance of that model. All models implement ExcursionModel.

        Returns:
            self (:obj:'ExcursionModel'): ExcursionModel instance.
        """

        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        for key, value in params.items():
            setattr(self, key, value)

        return self
