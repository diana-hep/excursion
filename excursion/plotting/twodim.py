import numpy as np
from ..utils import (
    point_entropy,
    point_entropy_gpytorch,
    mesh2points,
    points2mesh,
    values2mesh,
)
import torch

import matplotlib.pyplot as plt
from matplotlib import gridspec

def getminmax(ndarray):
    return np.min(ndarray), np.max(ndarray)


def plot_current_estimate(
    ax, gp, X, y, prediction, scandetails, funcindex, batchsize=1
):
    thresholds = scandetails.thresholds
    xv, yv = scandetails.plotG
    truthv = scandetails.truth_functions[funcindex](scandetails.plotX)
    truthv = values2mesh(truthv, scandetails.plot_rangedef, scandetails.invalid_region)

    vmin, vmax = getminmax(prediction[~np.isnan(prediction)])

    ax.contourf(xv, yv, prediction, np.linspace(vmin, vmax, 100))
    ax.contour(xv, yv, prediction, thresholds, colors="white", linestyles="solid")
    ax.contour(xv, yv, truthv, thresholds, colors="white", linestyles="dotted")
    ax.scatter(X[:-batchsize, 0], X[:-batchsize, 1], s=20)

    ax.scatter(
        X[:-batchsize, 0],
        X[:-batchsize, 1],
        s=20,
        c=y[:-batchsize],
        edgecolor="white",
        vmin=vmin,
        vmax=vmax,
    )
    ax.scatter(X[-batchsize:, 0], X[-batchsize:, 1], s=20, c="r")

    ax.set_xlim(*scandetails.plot_rangedef[0][:2])
    ax.set_ylim(*scandetails.plot_rangedef[1][:2])


def plot_current_entropies(ax, gp, X, entropies, scandetails, batchsize=1):
    thresholds = scandetails.thresholds
    xv, yv = scandetails.plotG

    vmin, vmax = getminmax(entropies)

    entropies = values2mesh(
        entropies, scandetails.plot_rangedef, scandetails.invalid_region
    )
    ax.contourf(xv, yv, entropies, np.linspace(vmin, vmax, 100))

    for truth_func in scandetails.truth_functions:
        truthv = truth_func(scandetails.plotX)
        truthv = values2mesh(
            truthv, scandetails.plot_rangedef, scandetails.invalid_region
        )
        ax.contour(xv, yv, truthv, thresholds, colors="white", linestyles="dotted")
    ax.scatter(X[:-batchsize, 0], X[:-batchsize, 1], s=20, c="w")
    ax.scatter(X[-batchsize:, 0], X[-batchsize:, 1], s=20, c="r")
    ax.set_xlim(*scandetails.plot_rangedef[0][:2])
    ax.set_ylim(*scandetails.plot_rangedef[1][:2])


def plot(axarr, gps, X, y_list, scandetails, batchsize=1):
    newX = scandetails.plotX

    mu_stds = []
    for i, (gp, y) in enumerate(zip(gps, y_list)):
        prediction, prediction_std = gp.predict(newX, return_std=True)
        mu_stds.append([prediction, prediction_std])

        prediction = values2mesh(
            prediction, scandetails.plot_rangedef, scandetails.invalid_region
        )
        prediction_std = values2mesh(
            prediction_std, scandetails.plot_rangedef, scandetails.invalid_region
        )

        axarr[i].set_title("GP #{}".format(i))
        plot_current_estimate(
            axarr[i],
            gp,
            X,
            y,
            prediction,
            scandetails,
            funcindex=i,
            batchsize=batchsize,
        )

    entropies = point_entropy(mu_stds, scandetails.thresholds)
    axarr[-1].set_title("Entropies")
    plot_current_entropies(
        axarr[-1], gp, X, entropies, scandetails, batchsize=batchsize
    )


def plot_GP(ax, gp, testcase, device, dtype, batchsize=1, entropy=False, **kwargs):
    """
    Plot GP posterior fit to data with the option of plotting side by side acquisition function
    """
    try:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8))
    except KeyError:
        print("I didnt get an axis")
    

    X_train = gp.train_inputs[0]
    y_train = gp.train_targets

    xv, yv = testcase.plot_meshgrid
    thresholds = testcase.thresholds

    # true function + thresholds
    X_plot = torch.Tensor(testcase.X_plot).to(device, dtype)
    truthv = testcase.true_functions[0](X_plot)
    truthv = truthv.to(device, dtype)
    truthv = values2mesh(truthv, testcase.rangedef, testcase.invalid_region)
    line0 = ax0.contour(
        xv,
        yv,
        truthv,
        thresholds,
        colors="white",
        linestyles="dotted",
        label="true contour",
    )

    ##mean
    gp.eval()
    likelihood = gp.likelihood
    likelihood.eval()
    prediction = likelihood(gp(X_plot))

    prediction = values2mesh(
        prediction.mean.detach().cpu().numpy(),
        testcase.rangedef,
        testcase.invalid_region,
    )

    vmin, vmax = getminmax(prediction[~np.isnan(prediction)])

    contour = ax0.contourf(xv, yv, prediction, np.linspace(vmin, vmax, 100))
    line1 = ax0.contour(
        xv, yv, prediction, thresholds, colors="white", linestyles="solid"
    )

    ##train points
    old_points = ax0.scatter(
        X_train[:-batchsize, 0].cpu(),
        X_train[:-batchsize, 1].cpu(),
        s=20,
        edgecolor="white",
        label="true sample",
    )
    new_point = ax0.scatter(
        X_train[-batchsize:, 0].cpu(),
        X_train[-batchsize:, 1].cpu(),
        s=20,
        c="r",
        label="last added",
    )
    ax0.set_title("New points selected")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_xlim(*testcase.rangedef[0][:2])
    ax0.set_ylim(*testcase.rangedef[1][:2])
    cbar0=fig.colorbar(contour, ax=ax0, shrink=0.7)
    cbar0.ax.set_ylabel('mean GP', size=14, labelpad=20,  rotation=270)
    ax0.legend(loc=0)
    l0, _ = line0.legend_elements()
    l1, _ = line1.legend_elements()

    ax0.legend(
        [l0[0], l1[0], old_points, new_point],
        ["True excursion set (thr=0)", "Estimation", "Observed points", "Next point"],
        #loc="bottom center",
        bbox_to_anchor=(0.7, -0.1),
        ncol=2,
        facecolor="grey",
        framealpha=0.20,
    )

    #plot entropy now
    acq = kwargs["acq"]
    acq_type = kwargs["acq_type"]
    line1 = ax1.contour(
        xv,
        yv,
        truthv,
        thresholds,
        colors="red",
        linestyles="dotted",
    )
    entropies_mesh = values2mesh(
        acq.detach().cpu().numpy(),
        testcase.rangedef,
        testcase.invalid_region,
    )
    entropy_vmin, entropy_vmax = getminmax(entropies_mesh[~np.isnan(entropies_mesh)])
    entropies = ax1.contourf(xv, yv, entropies_mesh, cmap="Purples")

    #new_point = ax1.scatter(
    #    X_train[-batchsize:, 0].cpu(),
    #    X_train[-batchsize:, 1].cpu(),
    #    s=20,
    #    c="r",
    #    label="last added",
    #)
    cbar1 = fig.colorbar(entropies, ax=ax1, shrink=0.7)
    cbar1.ax.set_ylabel('acquisition function', labelpad=20, rotation=270, size=14)
    ax1.set_title(acq_type)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")


    return ax


def my_func(x):
    return



def plot_GP_init(ax, gp, testcase, device, dtype, batchsize=1):
    """
    Plot GP posterior fit to data with the option of plotting side by side acquisition function
    """

    X_train = gp.train_inputs[0]
    y_train = gp.train_targets

    xv, yv = testcase.plot_meshgrid
    thresholds = testcase.thresholds

    # true function + thresholds
    X_plot = torch.Tensor(testcase.X_plot).to(device, dtype)
    truthv = testcase.true_functions[0](X_plot)
    truthv = truthv.to(device, dtype)
    truthv = values2mesh(truthv, testcase.rangedef, testcase.invalid_region)
    line0 = ax.contour(
        xv,
        yv,
        truthv,
        thresholds,
        colors="white",
        linestyles="dotted",
        label="true contour",
    )

    ##mean
    gp.eval()
    likelihood = gp.likelihood
    likelihood.eval()
    prediction = likelihood(gp(X_plot))

    prediction = values2mesh(
        prediction.mean.detach().cpu().numpy(),
        testcase.rangedef,
        testcase.invalid_region,
    )

    vmin, vmax = getminmax(prediction[~np.isnan(prediction)])

    ax1 = ax.contourf(xv, yv, prediction, np.linspace(vmin, vmax, 100))
    line1 = ax.contour(
        xv, yv, prediction, thresholds, colors="white", linestyles="solid"
    )

    ##train points
    old_points = ax.scatter(
        X_train[ 0].cpu(),
        X_train[1].cpu(),
        s=20,
        edgecolor="white",
        label="true sample",
    )
    

    ax.xlabel("x")
    ax.ylabel("y")
    ax.xlim(*testcase.rangedef[0][:2])
    ax.ylim(*testcase.rangedef[1][:2])
    ax.colorbar(ax1)
    ax.legend(loc=0)
    l0, _ = line0.legend_elements()
    l1, _ = line1.legend_elements()

    ax.legend(
        [l0[0], l1[0], old_points],
        ["True excursion set (thr=0)", "Estimation", "Observed points"],
        #loc="bottom center",
        bbox_to_anchor=(1.10, -0.1),
        ncol=2,
        facecolor="grey",
        framealpha=0.20,
    )

    return ax



















