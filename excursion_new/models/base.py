from collections import defaultdict


class ExcursionModel(object):
    def fit_model(self, model, x, y):
        raise NotImplemented

    def set_params(self, **params):
        """
        Set the parameters of this initial point generator.
        Parameters
        ----------
        **params : dict
            Generator parameters.
        Returns
        -------
        self : object
            Generator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        for key, value in params.items():
            setattr(self, key, value)

        return self
