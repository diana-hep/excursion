#!/usr/bin/python

from matplotlib import pyplot as plotter

import os
import numpy as np
import datetime

import sys
sys.path.append("/ptmp/mpp/prieck/atlasDM/dileptonReso/madgraph5atlasval_complete/madgraph5atlasval/excursion/")

import excursion
import excursion.testcases.fast as scandetails
import excursion.optimize
import excursion.plotting.twodim as exPlot
import excursion.utils


def parabola2d(x, min_x1 = -1.0, min_x2 = -1.0, scale_x1 = 0.1, scale_x2 = 0.4):
    x1, x2 = x[:,0], x[:,1]
    return scale_x1 * ( x1 - min_x1 )**2 + scale_x2 * ( x2 - min_x2 )**2

#--------------------------------------------------

def plot_and_save(workdir, index, gps, X, y_list, scandetails):
    fig, axarr = plotter.subplots(1, 2, sharey=True)
    fig.set_size_inches(9.5, 3.5)
    plotter.title('Iteration {}'.format(index))
    exPlot.plot(axarr, gps, X, y_list, scandetails)
    plotter.tight_layout()
    plotter.savefig(os.path.join(workdir,'update_{}.png'.format(str(index).zfill(3))))
    plotter.show()

#--------------------------------------------------

def main():
    
    np.warnings.filterwarnings('ignore')

    scandetails.truth_functions = [ parabola2d ]
    
    truthv = excursion.utils.values2mesh(
        parabola2d(scandetails.plotX),
        scandetails.plot_rangedef,
        scandetails.invalid_region
    )

    contours = [ truthv ]
    
    thresholds = [ 1.0 ]
    
    curve = plotter.contour( scandetails.plotG[0], scandetails.plotG[1], truthv, levels = thresholds, colors= [ 'b' ] )

    plotter.show()
#    plotter.savefig("truth_parabola2d_threshold_1p0.png", bbox_inches = 'tight')

    X, y_list, gps = excursion.optimize.init(scandetails, n_init = 3, seed = 1)
    

    workdir = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-2dmulti')
    os.mkdir(workdir)
    
    plot_and_save(workdir, 0, gps, X, y_list, scandetails)

    N_UPDATES = 10
    
    for iScan in range(N_UPDATES):
        if iScan > 0:
            gps = [ excursion.get_gp(X, y_list[0]) ]
        plot_and_save(workdir, iScan, gps, X, y_list, scandetails)
        X_new, acq_vals = excursion.optimize.gridsearch(gps, X, scandetails)
        print('Iteration {}. new x: {}'.format(iScan, X_new))
        y_new = [ scandetails.truth_functions[0](X_new) ]
        print('Iteration {}. Evaluted truth function to value: {}'.format(iScan, y_new))
        y_list[0] = np.concatenate( [y_list[0], y_new[0]] )
        X         = np.concatenate( [X, X_new] )
    
    

#--------------------------------------------------

if __name__ == "__main__":
    main()


