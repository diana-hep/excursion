#!/usr/bin/python

import sys
import os
import numpy as np
import datetime
import pickle

#
# for matplotlib (some problem preventing use of $PYTHONPATH)
#
sys.path.append("/afs/ipp-garching.mpg.de/common/soft/anaconda/amd64_generic/3/5.3.0/lib/python3.7/site-packages")

from matplotlib import pyplot as plotter

sys.path.append("/ptmp/mpp/prieck/atlasDM/dileptonReso/madgraph5atlasval_complete/madgraph5atlasval/excursion/")

import excursion
import excursion._optimize
import excursion.plotting.twodim as exPlot
import excursion.utils

import scan2d as scandetails
#import scan_gl_mzp as scandetails
import getMuLimitBatch


def plot_and_save(workdir, index, gps, X, y_list, scandetails, batchsize, doSaveInput = True):

    if doSaveInput:
        outFileXY = open(os.path.join(workdir, "xy_{}.pickle".format(index)), "wb")
        dirXY = { "X" : X,
                  "Y" : y_list,
                  "gps" : gps }
        pickle.dump(dirXY, outFileXY)
        outFileXY.close()
#        print("==================================================")
#        print("GP:")
#        print(gps[0].get_params())
#        print("==================================================")
    
    figure, axarr = plotter.subplots(1, 2, sharey=True)
    figure.set_size_inches(9.5, 3.5)
    plotter.title('Iteration {}'.format(index))
#    plotter.yscale("log")
    exPlot.plot(axarr, gps, X, y_list, scandetails, figure, batchsize)
    plotter.tight_layout()
    plotter.savefig(os.path.join(workdir,'update_{}.png'.format(str(index).zfill(3))), dpi = 300)
#    plotter.show()

#--------------------------------------------------

def main():
    
    np.warnings.filterwarnings('ignore')
    
    scandetails.truth_functions = [getMuLimitBatch.getMuLimitBatch]
    scandetails.thresholds      = [ 0.0 ]
    
    _n_init = 6
    X, y_list, gps = excursion.optimize.init(scandetails, n_init = _n_init, seed = 123)
    
    print("init X is {}".format(X))
    print("init y_list is {}".format(y_list))
    
    workdir = datetime.datetime.now().strftime('excursionTest_%Y-%m-%d-%H-%M-%S-2dmulti')
    os.mkdir(workdir)
    plot_and_save(workdir, 0, gps, X, y_list, scandetails, _n_init)
    
    N_UPDATES = 15
    
    for iScan in range(N_UPDATES):
        print("X is {}".format(X))
        print("y_list is {}".format(y_list))
        _batchsize = 6
        X_new, acq_vals = excursion.optimize.gridsearch(gps, X, scandetails, batchsize = _batchsize, resampling_frac = 1.0)
        print('Iteration {}. new x: {}'.format(iScan, X_new))
        y_new = [ scandetails.truth_functions[0](X_new) ]
        print('Iteration {}. Evaluted truth function to value: {}'.format(iScan, y_new))
        y_list[0] = np.concatenate( [y_list[0], y_new[0]] )
        X         = np.concatenate( [X, X_new] )
        gps       = [ excursion.get_gp(X, y_list[0]) ]
        plot_and_save(workdir, iScan + 1, gps, X, y_list, scandetails, _batchsize)
    
    outFile = open(os.path.join(workdir, "upperLimits.txt"), "w")
    for iPoint in range(len(y_list[0])):
        for iCoordinate in range(X.shape[-1]):
            outFile.write("{} ".format(X[iPoint][iCoordinate]))
        outFile.write(" -> {}\n".format(y_list[0][iPoint]))
    
#--------------------------------------------------

if __name__ == "__main__":
    main()

