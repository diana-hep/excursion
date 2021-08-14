import numpy as np
from .base import SampleGenerator


class RandomChoice(SampleGenerator):
    """Creates a set of n-dimensional points randomly chosen from the X_pointsgrid.
    The X_pointsgrid object represents a set of points in an n-dimensional space.

    """
    def __init__(self):
        None

    def generate(self, n_samples, meshgrid):
        indexs = np.random.choice(range(len(meshgrid)), size=n_samples, replace=False)
        X_init = meshgrid[indexs]
        return X_init