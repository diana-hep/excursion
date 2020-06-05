import numpy as np
import itertools
import utils


def confusion_matrix(gps, scandetails):
    thresholds = np.concatenate([[-np.inf], scandetails.thresholds, [np.inf]])
    diagX = utils.mesh2points(
        utils.mgrid(scandetails.plot_rangedef), scandetails.plot_rangedef[:, 2]
    )

    diagX = diagX[~scandetails.invalid_region(diagX)]

    return diagX
