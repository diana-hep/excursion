from skimage import measure
import numpy as np
from ..utils import (
    point_entropy,
    point_entropy_gpytorch,
    mesh2points,
    points2mesh,
    values2mesh,
    mgrid
)
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def getminmax(ndarray):
    return np.min(ndarray), np.max(ndarray)


def check_contour_3d(f, *args, **kw):
    try:
        f(*args, **kw)
        return True
    except Exception:
        return False


def contour_3d(v, rangedef, level, alpha=None, facecolors=None, edgecolors=None):
    verts, faces, normals, values = measure.marching_cubes(v, level=level, step_size=1)
    true = (
        rangedef[:, 0]
        + (rangedef[:, 1] - rangedef[:, 0]) * np.divide(1.0, rangedef[:, 2] - 1) * verts
    )
    mesh = Poly3DCollection(true[faces])

    if alpha:
        mesh.set_alpha(alpha)
    if facecolors:
        mesh.set_facecolors(facecolors)
    if edgecolors:
        mesh.set_edgecolors(edgecolors)
    return mesh


def plot_current_estimate(ax, gp, X, y, scandetails, funcindex, view_init=(70, -45)):
    denseGrid = mgrid(scandetails.plot_rangedef)
    denseX = mesh2points(denseGrid, scandetails.plot_rangedef[:, 2])

    prediction, prediction_std = gp.predict(denseX, return_std=True)
    ax.scatter(denseX[:, 0], denseX[:, 1], denseX[:, 2], c=prediction, alpha=0.05)

    for val, c in zip(scandetails.thresholds, ["r", "g", "y"]):
        vals = prediction.reshape(*map(int, scandetails.plot_rangedef[:, 2]))
        mesh = contour_3d(
            vals, scandetails.plot_rangedef, val, alpha=0.1, facecolors=c, edgecolors=c
        )
        ax.add_collection3d(mesh)

    truthy = scandetails.truth(denseX)
    for val, c in zip(scandetails.thresholds, ["k", "grey", "blue"]):
        vals = truthy.reshape(*map(int, scandetails.plot_rangedef[:, 2]))
        mesh = contour_3d(
            vals, scandetails.plot_rangedef, val, alpha=0.1, facecolors=c, edgecolors=c
        )
        ax.add_collection3d(mesh)

    scatplot = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c="r", s=100, alpha=0.2)

    # scatplot = ax.scatter(X[:,0],X[:,1],X[:,2], c = Y, alpha = 0.05, s = 200)
    ax.set_xlim(scandetails.plot_rangedef[0][0], scandetails.plot_rangedef[0][1])
    ax.set_ylim(scandetails.plot_rangedef[1][0], scandetails.plot_rangedef[1][1])
    ax.set_zlim(scandetails.plot_rangedef[2][0], scandetails.plot_rangedef[2][1])
    ax.view_init(*view_init)


def plot_GP(ax, gp, testcase, device, dtype, batchsize=1):
    """
    Plot GP posterior fit to data with the option of plotting side by side acquisition function
    """

    X_train = gp.train_inputs[0]
    y_train = gp.train_targets

    xv, yv, zv = testcase.plot_meshgrid
    thresholds = testcase.thresholds

    # true function + thresholds
    X_plot = testcase.X.to(device, dtype)
    truthv = testcase.true_functions[0](X_plot)
    truthv = truthv.to(device, dtype)
    truthv = values2mesh(truthv, testcase.rangedef, testcase.invalid_region)

    # true excursion set
    for val, c in zip(testcase.thresholds, ["k", "grey", "blue"]):
        vals = truthv.reshape(*map(int, testcase.rangedef[:, 2]))
        mesh = contour_3d(
            vals, testcase.rangedef, val, alpha=0.05, facecolors=c, edgecolors=c
        )
        ax.add_collection3d(mesh)

    # mean prediction
    gp.eval()
    likelihood = gp.likelihood
    likelihood.eval()
    prediction = likelihood(gp(X_plot))

    # plot heatmap mean
    prediction_mean = prediction.mean.detach().cpu()

    prediction_mean_mesh = values2mesh(
        prediction_mean, testcase.rangedef, testcase.invalid_region,
    )

    for val, c in zip(testcase.thresholds, ["r", "g", "y"]):
        vals = (prediction_mean_mesh).reshape(*map(int, testcase.rangedef[:, 2]))

        allow = check_contour_3d(
            contour_3d,
            vals,
            testcase.rangedef,
            val,
            alpha=0.1,
            facecolors=c,
            edgecolors=c,
        )
        print("allow ", allow)

        if allow:
            mesh = contour_3d(
                vals, testcase.rangedef, val, alpha=0.1, facecolors=c, edgecolors=c
            )
            ax.add_collection3d(mesh)

    # points of evaluation
    ax.scatter(
        X_train[:, 0].cpu(),
        X_train[:, 1].cpu(),
        X_train[:, 2].cpu(),
        c="r",
        s=70,
        alpha=0.8,
    )

    # limits
    ax.set_xlim(testcase.rangedef[0][0], testcase.rangedef[0][1])
    ax.set_ylim(testcase.rangedef[1][0], testcase.rangedef[1][1])
    ax.set_zlim(testcase.rangedef[2][0], testcase.rangedef[2][1])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # ax.view_init(*view_init)
