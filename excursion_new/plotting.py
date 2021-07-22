import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec


def plot_GP(gp, testcase, **kwargs):
    """
    Plot GP posterior fit to data with the option of plotting side by side acquisition function
    """

    X_train = gp.train_inputs[0].cpu()
    y_train = gp.train_targets.cpu()
    X_plot = torch.tensor(testcase.X, dtype=torch.float64, device=torch.device('cuda'))

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


