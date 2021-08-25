import numpy as np
import matplotlib.pyplot as plt
from .excursion import ExcursionResult
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_confusion_matrix(confusion_matrix, pct_correct: int):
    fig, ax = plt.subplots(1, figsize=(5, 5))
    plt.title("Confusion Matrix, "+str(pct_correct)+"% Accuracy")
    # plt.xticks(tick_marks, c, rotation=45)
    # plt.yticks(tick_marks, c)
    ax.imshow(confusion_matrix, cmap="binary")
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


def plot_1D(acq, train_y, train_X, plot_X, plot_G, rangedef, pred_mean, pred_cov, thresholds, next_x, true_y,
            invalid_region, func=None):
    fig = plt.figure(figsize=(15, 15))
    plot_X = plot_G[0]
    if acq is not None:
        gs = fig.add_gridspec(2, 1, height_ratios=[10, 5])
        fig_ax1 = fig.add_subplot(gs[0, :])
        fig_ax2 = fig.add_subplot(gs[1, :])
    else:
        gs = fig.add_gridspec(1, 1)
        fig_ax1 = fig.add_subplot(gs[0, :])

    fig_ax1.plot(plot_X, true_y, linestyle="dashed", color='k', label='True Function')

    min_X = np.min(plot_X)
    max_X = np.max(plot_X)
    fig_ax1.hlines(thresholds, min_X, max_X, colors="purple", label="threshold")

    # plot train points
    fig_ax1.plot(
        train_X,
        train_y,
        "k*",
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
    fig_ax1.set_xlim(*rangedef[0][:2])
    # fig_ax1.set_ylim(*rangedef[1][:2])
    # ax0.legend(loc="upper right")

    if acq is not None:
        # fig_ax2.set_xticks([])
        # eliminate -inf
        # mask = np.isfinite(acq)
        # acq = acq[mask]
        # X_plot = plot_X[mask]
        X_plot = plot_X
        # plot
        fig_ax2.plot(X_plot, acq, color="orange", label="Acquisition")
        # + str(acq_type))
        fig_ax2.set_xlabel("x")
        fig_ax2.set_ylabel("acq(x)")
        # fig_ax2.set_yscale("log")
        fig_ax2.axvline(next_x, c="red")
        fig_ax2.legend(loc="lower right")
        fig_ax2.set_xlim(*rangedef[0][:2])

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

    fig = plt.figure(figsize=(28, 8))
    if acq is not None:
        gs = fig.add_gridspec(1, 2)
        fig_ax1 = fig.add_subplot(gs[0, :1])
        fig_ax2 = fig.add_subplot(gs[0, 1:])
    else:
        gs = fig.add_gridspec(1, 1)
        fig_ax1 = fig.add_subplot(gs[0, :])

    xv, yv = plot_G

    line1 = fig_ax1.contour(xv, yv, true_y, thresholds, linestyles="dotted", colors='white')

    min_xv = np.min(pred_mean)
    max_xv = np.max(pred_mean)
    line2 = fig_ax1.contour(xv, yv, pred_mean, thresholds, colors="white", linestyles='solid')
    color_axis = fig_ax1.contourf(xv, yv, pred_mean, np.linspace(min_xv, max_xv, 100))

    # train points
    old_points = fig_ax1.scatter(
        train_X[:, 0],
        train_X[:, 1],
        s=20,
        edgecolor="white",
        label="Observed Truth Values",
    )
    new_point = None
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
    ax1_colorbar = fig.colorbar(color_axis, ax=fig_ax1, shrink=0.7)
    ax1_colorbar.ax.set_ylabel('mean GP', size=14, labelpad=10, rotation=270)
    fig_ax1.legend(loc=0)
    l1, _ = line1.legend_elements()
    l2, _ = line2.legend_elements()

    fig_ax1.legend(
        [l1[0], l2[0], old_points, new_point],
        ["True excursion set (thresholds=0)", "Estimation", "Observed points", "Next point"],
        # loc="bottom center",
        bbox_to_anchor=(0.7, -0.1),
        ncol=2,
        facecolor="grey",
        framealpha=0.20,
    )

    if acq is not None:
        # max_xv_ = np.max(acq)
        # min_xv_ = np.min(acq)
        acq = values2mesh(acq, plot_X, rangedef, invalid_region)
        # color_axis_ = fig_ax2.contourf(xv, yv, acq, np.linspace(min_xv_, max_xv_, 100))
        color_axis_ = fig_ax2.contourf(xv, yv, acq, cmap="Purples")
        # plot truth
        line_ = fig_ax2.contour(xv, yv, true_y, thresholds, linestyles="dotted", colors='r')
        # plot train points
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

        ax2_colorbar = fig.colorbar(color_axis_, ax=fig_ax2, shrink=0.7)
        ax2_colorbar.ax.set_ylabel('Acquisition Function', size=14, labelpad=10, rotation=270)
        fig_ax2.set_xlabel("x")
        fig_ax2.set_ylabel("y")
        fig_ax2.set_xlim(*rangedef[0][:2])
        fig_ax2.set_ylim(*rangedef[1][:2])
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


def contour_3d(v, rangedef, level, alpha=None, facecolors=None, edgecolors=None):
    verts, faces, normals, values = measure.marching_cubes(v, level=level, step_size=1)
    true = (rangedef[:, 0] + (rangedef[:, 1] - rangedef[:, 0]) * np.divide(1.0, rangedef[:, 2] - 1) * verts)
    mesh = Poly3DCollection(true[faces])
    if alpha: mesh.set_alpha(alpha)
    if facecolors: mesh.set_facecolors(facecolors)
    if edgecolors: mesh.set_edgecolors(edgecolors)
    return mesh


def plot_3D(acq, train_y, train_X, plot_X, plot_G, rangedef, pred_mean, pred_cov, thresholds, next_x, true_y,
            invalid_region, func=None):

    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, :], projection='3d')

    ax.scatter(plot_X[:, 0], plot_X[:, 1], plot_X[:, 2], c=pred_mean, alpha=0.05)

    for val, c in zip(thresholds, ["r", "g", "y"]):
        vals = pred_mean.reshape(*map(int, rangedef[:, 2]))
        mesh = contour_3d(
            vals, rangedef, val, alpha=0.1, facecolors=c, edgecolors=c
        )
        ax.add_collection(mesh)

    for val, c in zip(thresholds, ["k", "grey", "blue"]):
        vals = true_y.reshape(*map(int, rangedef[:, 2]))
        mesh = contour_3d(
            vals, rangedef, val, alpha=0.1, facecolors=c, edgecolors=c
        )
        ax.add_collection(mesh)

    scatplot = ax.scatter(train_X[:, 0], train_X[:, 1], train_X[:, 2], c="r", s=100, alpha=0.2)

    # scatplot = ax.scatter(train_X[:,0],train_X[:,1],train_X[:,2], c = Y, alpha = 0.05, s = 200)
    ax.set_xlim(rangedef[0][0], rangedef[0][1])
    ax.set_ylim(rangedef[1][0], rangedef[1][1])
    ax.set_zlim(rangedef[2][0], rangedef[2][1])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(*(70, -45))
    plt.show()


def plot_4D(acq, train_y, train_X, plot_X, plot_G, rangedef, pred_mean, pred_cov, thresholds, next_x, true_y,
            invalid_region, func=None):
    return


plot_n = {1: plot_1D,
          2: plot_2D,
          3: plot_3D,
          4: plot_4D,
          5: plot_4D}


def plot(result: ExcursionResult, show_confusion_matrix=False):
    if result is None or not result.train_y:
        raise ValueError("Result is not yet defined! Cannot plot this yet. First try calling ask-and-tell. "
                         "Jump start must be false.")
    if show_confusion_matrix:
        plot_confusion_matrix(*result.get_diagnostic())
    try:
        plt.clf()
        return plot_n[result.ndim](acq=result.acq_vals[-1], train_y=result.train_y[-1], train_X=result.train_X[-1],
                                   plot_X=result.X_pointsgrid, plot_G=result.X_meshgrid, rangedef=result.rangedef,
                                   pred_mean=result.mean[-1], pred_cov=result.cov[-1], thresholds=result.thresholds,
                                   next_x=result.next_x[-1], true_y=result.true_y, invalid_region=result.invalid_region)
    except ValueError as error_message:
        print(error_message)
        print("Going to skip plotting and keep training\n")
        return
