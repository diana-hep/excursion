import numpy as np
from utils import point_entropy
from utils import point_entropy_gpytorch
import torch

def plot(axarr, gps, X, y_list, scandetails, batchsize = 1):
    gp_axes = axarr[:len(y_list)]

    mu_stds = []
    for i,ax in enumerate(gp_axes):
        ax.scatter(X[:,0],y_list[i])
        for x in X[:-1,0]:
            ax.axvline(x, c = 'grey', alpha = 0.2)
        ax.axvline(X[-1,0], c = 'r')
        
        prediction, prediction_std = gps[i].predict(scandetails.plotX, return_std = True)
        mu_stds.append([prediction, prediction_std])

        ax.plot(scandetails.plotX[:,0],prediction)
        ax.fill_between(scandetails.plotX[:,0],prediction - prediction_std, prediction + prediction_std)

    
        for func in scandetails.truth_functions:
            ax.plot(scandetails.plotX.ravel(),func(scandetails.plotX).ravel(), c = 'k', linestyle = 'dashed')

        for thr in scandetails.thresholds:
            ax.hlines(thr, np.min(scandetails.plotX),np.max(scandetails.plotX), colors = 'grey')
        ax.set_xlim(np.min(scandetails.plotX),np.max(scandetails.plotX))
        ax.set_ylim(*scandetails.y_lim)

    entraxis = axarr[-1]
    entropies = point_entropy(mu_stds, scandetails.thresholds)
    entraxis.fill_between(scandetails.plotX[:,0],-entropies,entropies)
    entraxis.set_ylim(-1,1)
    entraxis.axhline(0, c = 'k')

    for x in X[:-1,0]:
        entraxis.axvline(x, c = 'grey', alpha = 0.2)
    entraxis.axvline(X[-1,0], c = 'r')


def plot_gpytorch(axarr, gps, likelihood, X, y_list, scandetails, batchsize = 1):
    gp_axes = axarr[:len(y_list)]
    mu_stds = []

    for ax in gp_axes:
        if(ax != axarr[-1]):
            ax.scatter(X,y_list)

        for x in X[:-1]:
            ax.axvline(x, c = 'grey', alpha = 0.2)
        ax.axvline(X[-1], c = 'r')

        if(ax != axarr[-1]):
            gps.eval()
            likelihood.eval()
            observed_pred = likelihood(gps(torch.tensor(scandetails.plotX)))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()

            # plot mean of the prediction function
            ax.plot(scandetails.plotX[:,0], observed_pred.mean.detach().numpy()) #scandetails.plotX[:,0] is a numpy array
            mu_stds.append(observed_pred)

            # plot confidence region
            ax.fill_between(scandetails.plotX[:,0], lower.detach().numpy(), upper.detach().numpy())

            for func in scandetails.truth_functions:
                ax.plot(scandetails.plotX.ravel(),func(scandetails.plotX).ravel(), c = 'k', linestyle = 'dashed')

            for thr in scandetails.thresholds:
                ax.hlines(thr, np.min(scandetails.plotX),np.max(scandetails.plotX), colors = 'grey')
            ax.set_xlim(np.min(scandetails.plotX),np.max(scandetails.plotX))
            ax.set_ylim(*scandetails.y_lim)

    entraxis = axarr[-1]
    entropies = point_entropy_gpytorch(mu_stds, scandetails.thresholds)
    entraxis.fill_between(scandetails.plotX[:,0],-entropies,entropies, color='steelblue')
    entraxis.set_ylim(-1,1)
    entraxis.axhline(0, c = 'k')

    for x in X[:-1]:
        entraxis.axvline(x, c = 'grey', alpha = 0.2)
    entraxis.axvline(X[-1], c = 'r')
