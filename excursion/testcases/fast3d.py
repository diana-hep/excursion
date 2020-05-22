import numpy
import numpy as np

from .. import utils

def truth(x):
    xv, yv, zv = x[:,0],x[:,1], x[:,2]
    '''more wiggles in physics case'''
    def xsec(xv,yv,zv):
        return (12*np.exp(-xv/2)+((0.1*np.cos(10*yv)))+((0.2*np.cos(15*xv))))*np.exp(-0.3*zv)

    def eff(xv,yv,zv):
        return np.tanh((1.3*xv-yv)+1)*1

    def stats(nevents):
        return (1-np.tanh((nevents-5)))/2.

    def analysis(xv,yv,zv):
        return stats(xsec(xv,yv,zv) * eff(xv,yv,zv))

    return 3*(np.log(analysis(xv,yv,zv)) - np.log(0.05))



truth_functions = [truth]

def invalid_region(x):
    return np.array([False]*len(x))

plot_rangedef = np.array([[0.0,1.5,41],[0.0,1.5,41],[0,1.5,41]])
plotG = utils.mgrid(plot_rangedef)
plotX = utils.mesh2points(plotG,plot_rangedef[:,2])

thresholds = [0.0]

acq_rd = np.array([[0.0,1.5,16],[0.0,1.5,16],[0.0,1.5,16]])
acqG = utils.mgrid(acq_rd)
acqX = utils.mesh2points(acqG,acq_rd[:,2])

mn_rd = np.array([[0.0,1.5,16],[0,1.5,16],[0.0,1.5,16]])
mnG   = utils.mgrid(mn_rd)
meanX  = utils.mesh2points(mnG,mn_rd[:,2])
