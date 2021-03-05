import numpy as np
from ..utils import point_entropy, values2mesh, values2mesh_masked

def getminmax(ndarray):
    return np.min(ndarray), np.max(ndarray)

def plot_current_estimate(ax, gp, X, y, prediction, scandetails, funcindex, batchsize = 1, evaluate_truth = False):
    thresholds = scandetails.thresholds
    xv, yv = scandetails.plotG

    if evaluate_truth:
        truthv = scandetails.functions[funcindex](scandetails.plotX)
        truthv = values2mesh_masked(truthv, scandetails.plot_rangedef, scandetails.invalid_region)

    vmin, vmax = getminmax(prediction[~np.isnan(prediction)])

    ax.contourf(xv, yv, prediction, np.linspace(vmin, vmax, 100))
    ax.contour(xv, yv, prediction,thresholds, colors='white',linestyles='solid')
    if evaluate_truth:
        ax.contour(xv, yv, truthv,thresholds, colors='white', linestyles='dotted')
    ax.scatter(X[:-batchsize, 0], X[:-batchsize, 1], s=20)

    if batchsize:
        ax.scatter(X[:-batchsize, 0], X[:-batchsize, 1], s=20, c=y[:-batchsize], edgecolor='w',vmin=vmin, vmax=vmax)
        ax.scatter(X[-batchsize:, 0], X[-batchsize:, 1], s=20, c='r')
    else:
        ax.scatter(X[:, 0], X[:, 1], s=20, c=y[:], edgecolor='w',vmin=vmin, vmax=vmax)

    ax.set_xlim(*scandetails.plot_rangedef[0][:2])
    ax.set_ylim(*scandetails.plot_rangedef[1][:2])

def plot_current_entropies(ax, gp, X, entropies, scandetails, batchsize=1, evaluate_truth = False):
    thresholds = scandetails.thresholds
    xv, yv = scandetails.plotG

    vmin, vmax = getminmax(entropies[~np.isnan(entropies)])


    entropies = values2mesh_masked(entropies, scandetails.plot_rangedef, scandetails.invalid_region)
    ax.contourf(xv, yv, entropies, np.linspace(vmin, vmax, 100))

    if evaluate_truth:
        for truth_func in  scandetails.functions:
            truthv = truth_func(scandetails.plotX)
            truthv = values2mesh_masked(truthv, scandetails.plot_rangedef, scandetails.invalid_region)
            ax.contour(xv, yv, truthv, thresholds, colors='white', linestyles='dotted')
    if batchsize:
        ax.scatter(X[:-batchsize, 0], X[:-batchsize, 1], s=20, c='w')
        ax.scatter(X[-batchsize:, 0], X[-batchsize:, 1], s=20, c='r')
    else:
        ax.scatter(X[:, 0], X[:, 1], s=20, c='w')
    ax.set_xlim(*scandetails.plot_rangedef[0][:2])
    ax.set_ylim(*scandetails.plot_rangedef[1][:2])

def plot(axarr, gps, X, y_list, scandetails, batchsize = 1, evaluate_truth = False):
    newX = scandetails.plotX

    mu_stds = []
    for i,(gp,y) in enumerate(zip(gps,y_list)):
        prediction, prediction_std = gp.predict(newX, return_std=True)
        mu_stds.append([prediction, prediction_std])

        prediction = values2mesh_masked(
            prediction,
            scandetails.plot_rangedef,
            scandetails.invalid_region
        )
        prediction_std = values2mesh_masked(
            prediction_std,
            scandetails.plot_rangedef,
            scandetails.invalid_region
        )

        axarr[i].set_title('GP #{}'.format(i))
        
        plot_current_estimate(
            axarr[i], gp, X, y,
            prediction,
            scandetails,
            funcindex=i,
            batchsize = batchsize,
            evaluate_truth=evaluate_truth
        )

    entropies = point_entropy(mu_stds, scandetails.thresholds)
    axarr[-1].set_title('Entropies')
    plot_current_entropies(
        axarr[-1],
        gp, X, entropies, scandetails,
        batchsize = batchsize,
        evaluate_truth=evaluate_truth
    )
