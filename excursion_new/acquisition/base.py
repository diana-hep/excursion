from collections import defaultdict


class AcquisitionFunction(object):
    def acquire(self, gp, thresholds, meshgrid):
        raise NotImplemented

    def set_params(self, **params):
        """
        Set the parameters of this acquisition function.
        Parameters
        ----------
        **params : dict
            Function parameters.
        Returns
        -------
        self : object
            Function instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        for key, value in params.items():
            setattr(self, key, value)

        return self