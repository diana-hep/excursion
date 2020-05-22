import numpy as np
from .. import utils

def truth(x):
    xv, yv = x[:,0],x[:,1]
    '''more wiggles in physics case'''
    def xsec(xv,yv):
        return 12*np.exp(-xv/2)+((0.1*np.cos(10*yv+1)))+((0.2*np.cos(15*xv)))

    def eff(xv,yv):
        return np.tanh((1.3*xv-yv)+1)

    def stats(nevents):
        return (1-np.tanh((nevents-5)))/2.

    def analysis(xv,yv):
        return stats(xsec(xv,yv) * eff(xv,yv))

    return 3*(np.log(analysis(xv,yv)) - np.log(0.05))



truth_functions = [truth]

def invalid_region(x):
    return np.array([False]*len(x))

plot_rangedef = np.array([[0.0,1.5,101],[0.0,1.5,101]])
plotG = utils.mgrid(plot_rangedef)
plotX = utils.mesh2points(plotG,plot_rangedef[:,2])

thresholds = [0.0]

acq_rd = np.array([[0.0,1.5,41],[0.0,1.5,41]])
acqG = utils.mgrid(acq_rd)
acqX = utils.mesh2points(acqG,acq_rd[:,2])

mn_rd = np.array([[0.0,1.5,41],[0,1.5,41]])
mnG   = utils.mgrid(mn_rd)
meanX  = utils.mesh2points(mnG,mn_rd[:,2])
