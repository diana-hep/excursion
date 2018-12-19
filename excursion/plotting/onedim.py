import numpy as np
from ..utils import point_entropy

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

    
        for func in scandetails.functions:
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
