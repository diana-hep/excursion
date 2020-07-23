from skimage import measure
import numpy as np
from ..utils import (
    point_entropy,
    point_entropy_gpytorch,
    mesh2points,
    points2mesh,
    values2mesh,
)
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D



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
    denseGrid = utils.mgrid(scandetails.plot_rangedef)
    denseX = utils.mesh2points(denseGrid, scandetails.plot_rangedef[:, 2])

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
    X_plot = torch.Tensor(testcase.X_plot).to(device, dtype)
    truthv = testcase.true_functions[0](X_plot)
    truthv = truthv.to(device, dtype)
    truthv = values2mesh(truthv, testcase.rangedef, testcase.invalid_region)

    # mean prediction
    #prediction, prediction_std = gp.predict(denseX, return_std=True)
    gp.eval()
    likelihood = gp.likelihood
    likelihood.eval()
    prediction = likelihood(gp(X_plot))

    #plot heatmap mean
    ax.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2], c=prediction.mean.detach().to(device,dtype).numpy(), alpha=0.02)

    #plot excursion set estimation
    prediction_mean = values2mesh(
        prediction.mean.detach().to(device,dtype).numpy(),
        testcase.rangedef,
        testcase.invalid_region,
    )

    for val, c in zip(testcase.thresholds, ["r", "g", "y"]):
        vals = (prediction_mean).reshape(*map(int, testcase.rangedef[:, 2]))
        mesh = contour_3d(
            vals, testcase.rangedef, val, alpha=0.1, facecolors=c, edgecolors=c
        )
        ax.add_collection3d(mesh)


    # true excursion set
    for val, c in zip(testcase.thresholds, ["k", "grey", "blue"]):
        vals = truthv.reshape(*map(int, testcase.rangedef[:, 2]))
        mesh = contour_3d(
            vals, testcase.rangedef, val, alpha=0.1, facecolors=c, edgecolors=c
        )
        ax.add_collection3d(mesh)

    # points of evaluation
    ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c="r", s=100, alpha=0.6)

    # limits
    ax.set_xlim(testcase.rangedef[0][0], testcase.rangedef[0][1])
    ax.set_ylim(testcase.rangedef[1][0], testcase.rangedef[1][1])
    ax.set_zlim(testcase.rangedef[2][0], testcase.rangedef[2][1])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    #ax.view_init(*view_init)
