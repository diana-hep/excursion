import numpy as np
import torch


def mesh2points(grid, npoints_tuple):
    ndim = len(npoints_tuple)
    X = np.moveaxis(grid, 0, ndim).reshape(int(np.product(npoints_tuple)), ndim)
    return X


def points2mesh(X, npoints_tuple):
    ndim = len(npoints_tuple)
    grid = np.moveaxis(X.reshape(*(npoints_tuple + [ndim,])), ndim, 0)
    return grid


def mgrid(rangedef):
    _rangedef = np.array(rangedef, dtype="complex128")
    slices = [slice(*_r) for _r in _rangedef]
    return np.mgrid[slices]


def values2mesh(values, rangedef, invalid, invalid_value=np.nan):
    grid = mgrid(rangedef)
    allX = mesh2points(grid, rangedef[:, 2])
    allv = np.zeros(len(allX))
    inv = invalid(allX)

    if torch.cuda.is_available() and type(values) == torch.Tensor:
        allv[~inv] = values.cpu()
    else:
        allv[~inv] = values

    if np.any(inv):
        allv[inv] = invalid_value
    return allv.reshape(*map(int, rangedef[:, 2]))
