import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from .excursion import ExcursionResult

import numpy as np
from ..utils import point_entropy


def plot_test(acq, gps, X, train_y, train_X, plot_X, pred_mean, pred_cov, thresholds, next_x, func=None):
    axes = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
    ax0 = plt.subplot(axes[0])
    ax1 = plt.subplot(axes[1])

    ax0.plot(plot_X, func(plot_X), linestyle="dashed", color='k', label='True Function')

    min_X = torch.min(plot_X)
    max_X = torch.max(plot_X)
    ax0.hlines(thresholds, min_X, max_X, colors="purple", label="threshold")

    ##train points
    ax0.plot(
        train_X,
        train_y,
        "k*",
        color="black",
        label="samples",
        markersize=10,
    )
    for x in train_X:
        ax0.axvline(x, alpha=0.2, color="grey")

    ax0.axvline(next_x, c="red", label="new evaluation")
    ax0.plot(plot_X, pred_mean, color="blue", label="mean")

    ##variance
    for i in range(1, 6):
        ax0.fill_between(
            plot_X,
            pred_mean + i * pred_cov ** 0.5,
            pred_mean - i * pred_cov ** 0.5,
            color="steelblue",
            alpha=0.6 / i,
            label=str(i) + "sigma",
        )

    ax0.set_xlabel("x")
    ax0.set_ylabel("f(x)")
    ax0.set_ylim(10, 10)
    ax0.legend(loc="upper right")

    #        ax1.set_xticks([], [])
    ax1.set_xticks([])
    # eliminate -inf
    mask = np.isfinite(acq)
    acq = acq[mask]
    X_plot = plot_X[mask]
    # plot
    ax1.plot(X_plot, acq, color="orange", label="MES")
    # + str(acq_type))
    ax1.set_xlabel("x")
    ax1.set_ylabel("acq(x)")
    # if acq_type == "MES":
    ax1.set_yscale("log")
    ax1.axvline(next_x, c="red")

    # ax1.legend(vertical, label="maximum")

    ax1.legend(loc="lower right")



def plot(axarr, gps, X, y_list, scandetails, ):
    gp_axes = axarr[: len(y_list)]

    mu_stds = []
    for i, ax in enumerate(gp_axes):
        ax.scatter(X[:, 0], y_list[i])
        for x in X[:-1, 0]:
            ax.axvline(x, c="grey", alpha=0.2)
        ax.axvline(X[-1, 0], c="r")

        prediction, prediction_std = gps[i].predict(scandetails.plotX, return_std=True)
        mu_stds.append([prediction, prediction_std])

        ax.plot(scandetails.plotX[:, 0], prediction)
        ax.fill_between(
            scandetails.plotX[:, 0],
            prediction - prediction_std,
            prediction + prediction_std,
        )

        for func in scandetails.truth_functions:
            ax.plot(
                scandetails.plotX.ravel(),
                func(scandetails.plotX).ravel(),
                c="k",
                linestyle="dashed",
            )

        for thr in scandetails.thresholds:
            ax.hlines(
                thr, np.min(scandetails.plotX), np.max(scandetails.plotX), colors="grey"
            )
        ax.set_xlim(np.min(scandetails.plotX), np.max(scandetails.plotX))
        ax.set_ylim(*scandetails.y_lim)

    # entraxis = axarr[-1]
    # entropies = point_entropy(mu_stds, scandetails.thresholds)
    # entraxis.fill_between(scandetails.plot_X[:, 0], -entropies, entropies)
    # entraxis.set_ylim(-1, 1)
    # entraxis.axhline(0, c="k")

    for x in X[:-1, 0]:
        entraxis.axvline(x, c="grey", alpha=0.2)
    entraxis.axvline(X[-1, 0], c="r")




def plot_GP(gp, testcase: ExcursionResult, **kwargs):
    """
    Plot GP posterior fit to data with the option of plotting_ side by side acquisition function
    """

    X_train = gp.train_inputs[0].cpu()
    y_train = gp.train_targets.cpu()
    X_plot = torch.tensor(testcase.plot_X, dtype=torch.float64, device=torch.device('cuda'))

    ##mean
    likelihood = gp.likelihood
    likelihood.eval()
    gp.eval()
    prediction = likelihood(gp(X_plot))

    X_plot = torch.from_numpy(testcase.X_plot)
    variance = prediction.variance.cpu()
    mean = prediction.mean.cpu()

    if len(kwargs) == 1:

        fig = plt.figure(figsize=(12, 7))

        # true function + thresholds
        for func in testcase.true_functions:
            plt.plot(
                X_plot,
                func(X_plot),
                linestyle="dashed",
                color="black",
                label="true function",
            )

        for thr in testcase.thresholds:
            min_X = torch.min(testcase.plot_X)
            max_X = torch.max(testcase.plot_X)
            plt.hlines(thr, min_X, max_X, colors="purple", label="threshold")

        # GP plot
        ##train points
        plt.plot(X_train, y_train, "k*", color="black", label="samples", markersize=10)
        for x in X_train:
            plt.axvline(x, alpha=0.2, color="grey")


        plt.plot(X_plot, mean.detach(), color="blue", label="mean")

        ##variance
        for i in [1, 2, 3, 4, 5]:
            plt.fill_between(
                X_plot[:],
                mean.detach().numpy()
                + i * variance.detach().numpy() ** 0.5,
                mean.detach().numpy()
                - i * variance.detach().numpy() ** 0.5,
                color="steelblue",
                alpha=0.6 / i ** 1.5,
                label=str(i) + "sigma",
            )

        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend(loc=0)
        # plt.show()

    else:

        acq = kwargs["acq"].cpu()
        acq_type = kwargs["acq_type"]
        xnew = kwargs["x_new"].cpu()

        # axis
        try:
            ax0 = plt.subplot(kwargs["ax0"])
            ax1 = plt.subplot(kwargs["ax1"])
        except KeyError:
            print("I didnt get an axis")
            fig = plt.figure(figsize=(12, 7))
            axes = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
            ax0 = plt.subplot(axes[0])
            ax1 = plt.subplot(axes[1])

        # GP plot
        # true function + thresholds

        for func in testcase.true_functions:
            ax0.plot(
                X_plot,
                func(X_plot),
                linestyle="dashed",
                color="black",
                label="true function",
            )

        for thr in testcase.thresholds:
            min_X = torch.min(testcase.X)
            max_X = torch.max(testcase.X)
            ax0.hlines(thr, min_X, max_X, colors="purple", label="threshold")

        ##train points
        ax0.plot(
            X_train.detach().numpy(),
            y_train.detach().numpy(),
            "k*",
            color="black",
            label="samples",
            markersize=10,
        )
        for x in X_train:
            ax0.axvline(x, alpha=0.2, color="grey")

        ax0.axvline(X_train[-1, 0], c="red", label="new evaluation")

        ##mean
        # gp.eval()
        # likelihood = gp.likelihood
        # likelihood.eval()
        # prediction = likelihood(gp(testcase.X))
        ax0.plot(X_plot, mean.detach(), color="blue", label="mean")

        ##variance
        for i in range(1, 6):
            ax0.fill_between(
                X_plot,
                mean.detach().numpy() + i * variance.detach().numpy() ** 0.5,
                mean.detach().numpy() - i * variance.detach().numpy() ** 0.5,
                color="steelblue",
                alpha=0.6 / i,
                label=str(i) + "sigma",
            )

        ax0.set_xlabel("x")
        ax0.set_ylabel("f(x)")
        ax0.set_ylim(-2, 10)
        ax0.legend(loc="upper right")

        # ACQ plot

#        ax1.set_xticks([], [])
        ax1.set_xticks([])
        # eliminate -inf
        acq = acq.detach().numpy()
        mask = np.isfinite(acq)
        acq = acq[mask]
        X_plot = X_plot[mask]
        # plot
        ax1.plot(X_plot, acq, color="orange", label="EIG " + str(acq_type))
        ax1.set_xlabel("x")
        ax1.set_ylabel("acq(x)")
        if acq_type == "MES":
            ax1.set_yscale("log")

        for x in xnew:
            vertical = ax1.axvline(x, c="red")

        # ax1.legend(vertical, label="maximum")

        ax1.legend(loc="lower right")

        plt.subplots_adjust(hspace=0.0)
        # plt.show()

