import numpy as np

from .. import utils
from .. import ExcursionProblem

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


bounding_box = [[0,1.5],[0,1.5],[0,1.5]]
npoints = [75,75,75]
single_function = ExcursionProblem([truth],[0.0],ndim = 3, bounding_box = bounding_box, plot_npoints=npoints, n_acq=1000, n_mean=1000)
