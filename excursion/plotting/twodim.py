import numpy as np
from ..utils import point_entropy, mesh2points, points2mesh, values2mesh

def getminmax(ndarray):
    return np.min(ndarray), np.max(ndarray)

def plot_current_estimate(ax, gp, X, y, prediction, scandetails, funcindex, batchsize = 1):
    thresholds = scandetails.thresholds
    xv, yv = scandetails.plotG
    truthv = scandetails.truth_functions[funcindex](scandetails.plotX)
    truthv = values2mesh(truthv, scandetails.plot_rangedef, scandetails.invalid_region)

    vmin, vmax = getminmax(prediction[~np.isnan(prediction)])

    ax.contourf(xv, yv, prediction, np.linspace(vmin, vmax, 100))
    ax.contour(xv, yv, prediction,thresholds, colors='white',linestyles='solid')
    ax.contour(xv, yv, truthv,thresholds, colors='white', linestyles='dotted')
    ax.scatter(X[:-batchsize, 0], X[:-batchsize, 1], s=20)

    ax.scatter(X[:-batchsize, 0], X[:-batchsize, 1], s=20, c=y[:-batchsize], edgecolor='white',vmin=vmin, vmax=vmax)
    ax.scatter(X[-batchsize:, 0], X[-batchsize:, 1], s=20, c='r')

    ax.set_xlim(*scandetails.plot_rangedef[0][:2])
    ax.set_ylim(*scandetails.plot_rangedef[1][:2])

def plot_current_entropies(ax, gp, X, entropies, scandetails, batchsize=1):
    thresholds = scandetails.thresholds
    xv, yv = scandetails.plotG

    vmin, vmax = getminmax(entropies)


    entropies = values2mesh(entropies, scandetails.plot_rangedef, scandetails.invalid_region)
    ax.contourf(xv, yv, entropies, np.linspace(vmin, vmax, 100))

    for truth_func in  scandetails.truth_functions:
        truthv = truth_func(scandetails.plotX)
        truthv = values2mesh(truthv, scandetails.plot_rangedef, scandetails.invalid_region)
        ax.contour(xv, yv, truthv, thresholds, colors='white', linestyles='dotted')
    ax.scatter(X[:-batchsize, 0], X[:-batchsize, 1], s=20, c='w')
    ax.scatter(X[-batchsize:, 0], X[-batchsize:, 1], s=20, c='r')
    ax.set_xlim(*scandetails.plot_rangedef[0][:2])
    ax.set_ylim(*scandetails.plot_rangedef[1][:2])

def plot(axarr, gps, X, y_list, scandetails, batchsize = 1):
    newX = scandetails.plotX

    mu_stds = []
    for i,(gp,y) in enumerate(zip(gps,y_list)):
        prediction, prediction_std = gp.predict(newX, return_std=True)
        mu_stds.append([prediction, prediction_std])

        prediction = values2mesh(
            prediction,
            scandetails.plot_rangedef,
            scandetails.invalid_region
        )
        prediction_std = values2mesh(
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
            batchsize = batchsize
        )

    entropies = point_entropy(mu_stds, scandetails.thresholds)
    axarr[-1].set_title('Entropies')
    plot_current_entropies(
        axarr[-1],
        gp, X, entropies, scandetails,
        batchsize = batchsize
    )
