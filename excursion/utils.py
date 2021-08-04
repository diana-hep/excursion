import numpy as np
import torch
from scipy.stats import norm


def mesh2points(grid, npoints_tuple):
    ndim = len(npoints_tuple)
    X = np.moveaxis(grid, 0, ndim).reshape(int(np.product(npoints_tuple)), ndim)
    return X


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


def points2mesh(X, npoints_tuple):
    ndim = len(npoints_tuple)
    grid = np.moveaxis(X.reshape(*(npoints_tuple + [ndim,])), ndim, 0)
    return grid


def point_entropy(mu_stds, thresholds):
    thresholds = np.concatenate([[-np.inf], thresholds, [np.inf]])

    entropies = []
    for mu, std in mu_stds:
        entropy = 0
        for j in range(len(thresholds) - 1):
            p_within = norm(mu, std).cdf(thresholds[j + 1]) - norm(mu, std).cdf(
                thresholds[j]
            )
            p_within[p_within < 1e-9] = 1e-9
            p_within[p_within > 1 - 1e-9] = 1 - 1e-9
            entropy -= p_within * np.log(p_within)
        entropies.append(entropy)
    return np.mean(np.stack(entropies), axis=0)


def point_entropy_gpytorch(mu_stds, thresholds):
    thresholds = np.concatenate([[-np.inf], thresholds, [np.inf]])

    entropies = []
    for obs_pred in mu_stds:
        entropy = 0
        for j in range(len(thresholds) - 1):
            p_within = norm(
                obs_pred.mean.detach().numpy(), obs_pred.stddev.detach().numpy()
            ).cdf(thresholds[j + 1]) - norm(
                obs_pred.mean.detach().numpy(), obs_pred.stddev.detach().numpy()
            ).cdf(
                thresholds[j]
            )
            p_within[p_within < 1e-9] = 1e-9
            p_within[p_within > 1 - 1e-9] = 1 - 1e-9
            entropy -= p_within * np.log(p_within)
        entropies.append(entropy)
    return np.mean(np.stack(entropies), axis=0)

#
# def load_example(example):
#     testcase = None
#     if example == "1Dtoyanalysis":
#         testcase = importlib.import_module("excursion.unfactored.fast_1D")
#     elif example == "1D_test":
#         testcase = importlib.import_module("excursion.unfactored.1D_test")
#     elif example == "2D_test":
#         testcase = importlib.import_module("excursion.unfactored.2D_test")
#     elif example == "3D_test":
#         testcase = importlib.import_module("excursion.unfactored.3D_test")
#     elif example == "2Dtoyanalysis":
#         testcase = importlib.import_module("excursion.unfactored.fast_2D")
#     elif example == "darkhiggs":
#         testcase = importlib.import_module("excursion.unfactored.darkhiggs")
#     elif example == "checkmate":
#         testcase = importlib.import_module("excursion.unfactored.checkmate")
#     elif example == "3dfoursheets":
#         testcase = importlib.import_module("excursion.unfactored.toy3d_foursheets")
#     elif example == "3Dtoyanalysis":
#         testcase = importlib.import_module("excursion.unfactored.fast_3D")
#     elif example == "darkhiggs":
#         testcase = importlib.import_module("excursion.unfactored.darkhiggs")
#     elif example == "checkmate":
#         testcase = importlib.import_module("excursion.unfactored.checkmate")
#     elif example.startswith("parabola_"):
#         n = [int(s) for s in example if s.isdigit()][0]
#         # make_parabola_script(n)
#         testcase = importlib.import_module(
#             "excursion.unfactored.parabola_" + str(n) + "D"
#         )
#     # elif example == "parabola_1D":
#     #    testcase = importlib.import_module("excursion.unfactored.parabola_1D")
#     # elif example == "parabola_2D":
#     #    testcase = importlib.import_module("excursion.unfactored.parabola_2D")
#     # elif example == "parabola_3D":
#     #    testcase = importlib.import_module("excursion.unfactored.parabola_3D")
#     # elif example == "parabola_4D":
#     #    testcase = importlib.import_module("excursion.unfactored.parabola_4D")
#     else:
#         raise RuntimeError("unnkown test case")
#     return testcase
#
