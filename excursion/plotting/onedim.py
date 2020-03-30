import numpy as np
from excursion.utils import point_entropy
from excursion.utils import point_entropy_gpytorch
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec


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
    gp_axes = axarr[:len(y_list)] #IRINA
    mu_stds = []

    #GP posterior
    for ax in gp_axes:
        if(ax != axarr[-1]):
            ax.scatter(X,y_list)

        for x in X[:-1]:
            ax.axvline(x, c = 'grey', alpha = 0.2)
        ax.axvline(X[-1], c = 'r')

        if(ax != axarr[-1]):
            gps.eval()
            likelihood.eval()
            observed_pred = likelihood(gps( torch.tensor(scandetails.plotX, dtype=torch.float64) ))

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


    #point entropy
    entraxis = axarr[-1]
    entropies = point_entropy_gpytorch(mu_stds, scandetails.thresholds)
    entraxis.fill_between(scandetails.plotX[:,0],-entropies,entropies, color='steelblue')
    entraxis.set_ylim(-1,1)
    entraxis.axhline(0, c = 'k')

    for x in X[:-1]:
        entraxis.axvline(x, c = 'grey', alpha = 0.2)
    entraxis.axvline(X[-1], c = 'r')



def plot_GP(gp, testcase, **kwargs):
    """
    Plot GP posterior fit to data with the option of plotting side by side acquisition function
    """

    X_train = gp.train_inputs[0]
    y_train = gp.train_targets
    X_plot = testcase.plot_X

    
    if (len(kwargs)==0):

        fig = plt.figure(figsize=(12, 7))

        #true function + thresholds
        for func in testcase.true_functions:
            plt.plot(X_plot, func(X_plot), linestyle='dashed', color='black', label='true function')
        
        for thr in testcase.thresholds:
            min_X = torch.min(testcase.plot_X)
            max_X = torch.max(testcase.plot_X)
            plt.hlines(thr, min_X, max_X, colors = 'purple', label='threshold')


        #GP plot
        ##train points
        plt.plot(X_train, y_train,'k*', color='black', label='samples', markersize=10)
        for x in X_train:
            plt.axvline(x, alpha=0.2, color='grey')

        ##mean 
        gp.eval()
        likelihood = gp.likelihood
        likelihood.eval()
        prediction = likelihood(gp(X_plot))
        plt.plot(X_plot, prediction.mean.detach(), color='blue', label='mean')

        ##variance
        lower, upper = prediction.confidence_region()
        plt.fill_between(X_plot[:,0], lower.detach().numpy(), upper.detach().numpy(), color='steelblue', alpha=0.6, label='2sigma')

        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend(loc=0)
        plt.show()


    else:

        acq = kwargs['acq']

        #axis
        fig = plt.figure(figsize=(12, 7))
        axes = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
        #GP plot
        #true function + thresholds
        ax1 = plt.subplot(axes[1])
        for func in testcase.true_functions:
            ax1.plot(X_plot, func(X_plot), linestyle='dashed', color='black', label='true function')
        
        for thr in testcase.thresholds:
            min_X = torch.min(testcase.plot_X)
            max_X = torch.max(testcase.plot_X)
            ax1.hlines(thr, min_X, max_X, colors = 'purple', label='threshold')

        ##train points
        ax1.plot(X_train, y_train,'k*', color='black', label='samples', markersize=10)
        for x in X_train:
            ax1.axvline(x, alpha=0.2, color='grey')

        ax1.axvline(X_train[-1,0], c = 'r')

        ##mean 
        gp.eval()
        likelihood = gp.likelihood
        likelihood.eval()
        prediction = likelihood(gp(X_plot))
        ax1.plot(X_plot, prediction.mean.detach(), color='blue', label='mean')

        ##variance
        lower, upper = prediction.confidence_region()
        ax1.fill_between(X_plot[:,0], lower.detach().numpy(), upper.detach().numpy(), color='steelblue', alpha=0.6, label='2sigma')

        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.legend(loc='upper right')

        #ACQ plot
        ax0 = plt.subplot(axes[0])

        ax0.set_xticks([], [])
        #eliminate -inf
        acq = np.array(acq)
        X_plot = X_plot.numpy()
        mask = np.isfinite(acq)
        acq = acq[mask]
        X_plot = X_plot[mask]
        #plot
        ax0.plot(X_plot, acq ,color='orange', label='EIG PES')
        ax0.set_xlabel('x')
        ax0.set_ylabel('acq(x)')

        for x in X_train:
            ax0.axvline(x, alpha=0.2, color='grey')

        ax0.axvline(X_train[-1,0], c = 'r', label='new evaluation')
        ax0.legend(loc='lower right')

        fig.tight_layout()
        plt.subplots_adjust(hspace=.0)
        plt.show()

    
    



