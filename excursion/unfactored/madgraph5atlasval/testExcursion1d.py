#!/usr/bin/python

import sys
import numpy as np

sys.path.append("/afs/ipp-garching.mpg.de/common/soft/anaconda/amd64_generic/3/5.3.0/lib/python3.7/site-packages")

sys.path.append("/ptmp/mpp/prieck/atlasDM/dileptonReso/madgraph5atlasval_complete/madgraph5atlasval/excursion/")

import excursion
import excursion._optimize
import excursion.utils

import scan1d as scandetails
import getMuLimitFast1d

from matplotlib import pyplot as plt

plt.rcParams.update( { 'font.size' : 11 } )

#--------------------------------------------------

def plot(gps, X, y_list, xTruth, yTruth, scandetails, index):

    xMin = 0.0
    xMax = 1.0
    xp   = np.arange( xMin, xMax, ( xMax - xMin ) / 5e2  )
    xg   = xp.reshape(-1, 1)
    muFit, sigmaFit = gps[0].predict(xg, return_std = True)
    fig, axarr = plt.subplots(1, 2, figsize = plt.figaspect( 0.5 ))
    axarr[1].set_title("Exclusion Entropy")
    axarr[0].set_title("GP Regression")
    xTruthP = []
    for x in xTruth:
        xTruthP.append( x[0] )
    xTruthP = scandetails.mapXforPlotting(xTruthP)
    xpMass  = scandetails.mapXforPlotting(xp)
    XPMass  = scandetails.mapXforPlotting(X.reshape(1,-1)[0])
    h0, = axarr[0].plot(xTruthP, yTruth, ls = "-", c = "k", label = "True limit")
    h1  = axarr[0].scatter(XPMass, y_list[0], c="g", label = "Evaluated")
    h2, = axarr[0].plot(xpMass, muFit, c = "r", label = "GP mean")
    axarr[0].plot(xpMass, np.zeros(len(xpMass)), c= "b", ls = "--")
    h3  = axarr[0].fill_between(xpMass, muFit - sigmaFit, muFit + sigmaFit, alpha = 0.4, color = "k", label = "+/- 1 st. dev.")

    entropies = excursion.utils.point_entropy([[muFit, sigmaFit]], scandetails.thresholds)
    axarr[1].set_ylabel("Entropy [a.u.]")
    axarr[1].plot(xpMass, entropies, color = "b")

    xLabel = "$m_{Z'}$ [GeV]"
    axarr[0].set_xlabel(xLabel)
    axarr[1].set_xlabel(xLabel)
    axarr[0].set_ylabel("$\ln(\mu^{\mathrm{UL}})$")
    for ax in axarr:
        ax.set_xlim( np.min(xpMass), 6e3 ) # np.max(xpMass))
    axarr[0].set_ylim(-8.0, 5.0)
    axarr[0].legend(handles = [h0, h1, h2, h3], loc = 4, fontsize = 9)
#    plt.yscale('log')
    plt.sca(axarr[0])
    plt.text(0.8e3, 4, "Dilepton resonance limits, Run 2", fontsize = 9)
    plt.text(0.8e3, 2, "Axial-vector mediator, Dirac DM\n$g_q = g_{\ell} = 0.1, g_{\mathrm{DM}} = 1.0$\n$m_{\mathrm{DM}} = 500$ GeV", fontsize = 9)
#    plt.yticks( np.arange(-8, 8, 1) )
    plt.tight_layout()
    plt.savefig("dileptonLimits_1d_{}.png".format(index))
    
    

#--------------------------------------------------

def main():
    
    np.warnings.filterwarnings('ignore')

#    scandetails.truth_functions = [ getMuLimitBatch.getMuLimitBatch ]
#    scandetails.truth_functions = [ myTest.parabola2d ]
    scandetails.truth_functions = [getMuLimitFast1d.getMuLimitFast1d]
    scandetails.thresholds      = [ 0.0 ]

    xTruth = []
    for x in np.linspace(0.0, 1.0, 100):
        xTruth.append( [x] )
    yTruth = scandetails.truth_functions[0](xTruth)
    
    X, y_list, gps = excursion.optimize.init(scandetails, n_init = 3, seed = 123)
    plot(gps, X, y_list, xTruth, yTruth, scandetails, 0)
    
    print("init X is {}".format(X))
    print("init y_list is {}".format(y_list))
    
    N_UPDATES = 4
    
    for iScan in range(N_UPDATES):
        X_new, acq_vals = excursion.optimize.gridsearch(gps, X, scandetails, batchsize = 1, resampling_frac = 1.0)
        y_new = [ scandetails.truth_functions[0](X_new) ]
        y_list[0] = np.concatenate( [y_list[0], y_new[0]] )
        X         = np.concatenate( [X, X_new] )
        gps       = [ excursion.get_gp(X, y_list[0]) ]
        plot(gps, X, y_list, xTruth, yTruth, scandetails, iScan + 1)

        
        
#--------------------------------------------------

if __name__ == "__main__":
    main()

