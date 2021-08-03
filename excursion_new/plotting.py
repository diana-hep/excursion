import numpy as np
import matplotlib.pyplot as plt
from .excursion import ExcursionResult


def plot_confusion_matrix(confusion_matrix, pct_correct: int):
    plt.title("Confusion Matrix, "+str(pct_correct)+"% Accuracy")
    # plt.xticks(tick_marks, c, rotation=45)
    # plt.yticks(tick_marks, c)
    plt.imshow(confusion_matrix, cmap="binary")
    for i1 in range(confusion_matrix.shape[0]):
        for i2 in range(confusion_matrix.shape[1]):
            plt.text(
                i1,
                i2,
                confusion_matrix[i1][i2],
                ha="center",
                va="center",
                color="red",
            )
    plt.show()


def plot_2D(acq, train_y, train_X, plot_X, plot_G, rangedef, pred_mean, pred_cov, thresholds, next_x, true_y,
            invalid_region, func=None):

    def values2mesh(values, plot_X, plot_rangedef, invalid, invalid_value=np.nan):
        allv = np.zeros(len(plot_X))
        inv = invalid(plot_X)
        allv[~inv] = values
        if np.any(inv):
            allv[inv] = invalid_value
        return allv.reshape(*map(int, plot_rangedef[:, 2]))

    true_y = values2mesh(true_y, plot_X, rangedef, invalid_region)
    pred_mean = values2mesh(pred_mean, plot_X, rangedef, invalid_region)

    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 2)
    fig_ax1 = fig.add_subplot(gs[0, :1])
    fig_ax2 = fig.add_subplot(gs[0, 1:])

    xv, yv = plot_G

    line1 = fig_ax1.contour(xv, yv, true_y, thresholds, linestyle="dashed", color='white', label='True Contour')

    min_xv = np.min(pred_mean)
    max_xv = np.max(pred_mean)
    line2 = fig_ax1.contour(xv, yv, pred_mean, thresholds, colors="purple", label="threshold")
    color_axis = fig_ax1.contourf(xv, yv, pred_mean, np.linspace(min_xv, max_xv, 100))

    # train points
    old_points = fig_ax1.scatter(
        train_X[:, 0],
        train_X[:, 1],
        s=20,
        edgecolor="white",
        label="Observed Truth Values",
    )

    if next_x is not None:
        new_point = fig_ax1.scatter(
            next_x[:, 0],
            next_x[:, 1],
            s=20,
            c="r",
            label="New Observed Value",
        )

    fig_ax1.set_xlabel("x")
    fig_ax1.set_ylabel("y")
    fig_ax1.set_xlim(*rangedef[0][:2])
    fig_ax1.set_ylim(*rangedef[1][:2])
    fig.colorbar(color_axis, ax=fig_ax1)
    fig_ax1.legend(loc=0)
    l1, _ = line1.legend_elements()
    l2, _ = line2.legend_elements()

    fig_ax1.legend(
        [l1[0], l2[0], old_points, new_point],
        ["True excursion set (thresholds=0)", "Estimation", "Observed points", "Next point"],
        # loc="bottom center",
        bbox_to_anchor=(1.10, -0.1),
        ncol=2,
        facecolor="grey",
        framealpha=0.20,
    )

    if acq is not None:
        max_xv_ = np.max(acq)
        min_xv_ = np.min(acq)
        acq = values2mesh(acq, plot_X, rangedef, invalid_region)
        color_axis_ = fig_ax2.contourf(xv, yv, acq, np.linspace(min_xv_, max_xv_, 100))
        # plot truth
        line_ = fig_ax1.contour(xv, yv, true_y, thresholds, linestyle="dashed", color='white', label='True Contour')
        # plot
        # train points
        old_points_ = fig_ax2.scatter(
            train_X[:, 0],
            train_X[:, 1],
            s=20,
            edgecolor="white",
            label="Observed Truth Values",
        )

        if next_x is not None:
            new_point_ = fig_ax2.scatter(
                next_x[:, 0],
                next_x[:, 1],
                s=20,
                c="r",
                label="New Observed Value",
            )

        fig_ax2.set_xlabel("x")
        fig_ax2.set_ylabel("y")
        fig_ax2.set_xlim(*rangedef[0][:2])
        fig_ax2.set_ylim(*rangedef[1][:2])
        fig.colorbar(color_axis_, ax=fig_ax2)
        fig_ax2.legend(loc=0)
        l_, _ = line_.legend_elements()

        fig_ax2.legend(
            [l_[0], old_points_, new_point_],
            ["True excursion set (thresholds=0)", "Acquisition Value", "Observed points", "Next point"],
            # loc="bottom center",
            bbox_to_anchor=(1.10, -0.1),
            ncol=2,
            facecolor="grey",
            framealpha=0.20,
        )
        fig_ax2.legend(loc="lower right")

    plt.show()


def plot_1D(acq, train_y, train_X, plot_X, plot_G, rangedef, pred_mean, pred_cov, thresholds, next_x, true_y,
            invalid_region, func=None):
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(2, 1, height_ratios=[10, 5])
    fig_ax1 = fig.add_subplot(gs[0, :])
    fig_ax2 = fig.add_subplot(gs[1, :])
    plot_X = plot_G[0]

    fig_ax1.plot(plot_X, true_y, linestyle="dashed", color='k', label='True Function')

    min_X = np.min(plot_X)
    max_X = np.max(plot_X)
    fig_ax1.hlines(thresholds, min_X, max_X, colors="purple", label="threshold")

    # #train points
    fig_ax1.plot(
        train_X,
        train_y,
        "k*",
        color="black",
        label="samples",
        markersize=10,
    )
    for x in train_X:
        fig_ax1.axvline(x, alpha=0.2, color="grey")

    if next_x:
        fig_ax1.axvline(next_x, c="red", label="new evaluation")
    fig_ax1.plot(plot_X, pred_mean, color="blue", label="mean")

    # variance
    for i in range(1, 6):
        fig_ax1.fill_between(
            plot_X,
            pred_mean + 1*i * pred_cov ** 0.5,
            pred_mean - 1*i * pred_cov ** 0.5,
            color="darkslateblue",
            alpha=0.6 / i,
            label=str(i) + "sigma",
        )

    fig_ax1.set_xlabel("x")
    fig_ax1.set_ylabel("f(x)")
    fig_ax1.set_ylim(-6, 20)
    # ax0.legend(loc="upper right")

    if acq is not None:
        fig_ax2.set_xticks([])
        # eliminate -inf
        mask = np.isfinite(acq)
        acq = acq[mask]
        X_plot = plot_X[mask]
        # plot
        fig_ax2.plot(X_plot, acq, color="orange", label="MES")
        # + str(acq_type))
        fig_ax2.set_xlabel("x")
        fig_ax2.set_ylabel("acq(x)")
        # if acq_type == "MES":
        fig_ax2.set_yscale("log")
        fig_ax2.axvline(next_x, c="red")
        fig_ax2.legend(loc="lower right")

    plt.show()


def plot_3D(acq, train_y, train_X, plot_X, plot_G, rangedef, pred_mean, pred_cov, thresholds, next_x, true_y,
            invalid_region, func=None):
    return


plot_n = {1: plot_1D,
          2: plot_2D,
          3: plot_3D}


def plot(result: ExcursionResult, show_confusion_matrix=False):
    if show_confusion_matrix:
        plot_confusion_matrix(result.confusion_matrix, result.pct_correct)
    return plot_n[result.ndim](acq=result.acq, train_y=result.train_y, train_X=result.train_X, plot_X=result.plot_X,
                               plot_G=result.plot_G, rangedef=result.rangedef, pred_mean=result.mean,
                               pred_cov=result.cov, thresholds=result.thr, next_x=result.next_x, true_y=result.true_y,
                               invalid_region=result.invalid_region)
