import numpy as np
from .. import utils

def truth(X):
    x,y,z = X[:,0],X[:,1],X[:,2]
    return np.cos(5*x)*x**2+6*np.cos(5*z)*y**2+4*z**2-1.0

thresholds = [0.0]
truth_functions = [truth]

plot_rangedef = np.array([[-1,1,41],[-1,1,21],[-1,1,41]])

acq_rd = np.array([[-1,1,15],[-1,1,15],[-1,1,15]])
acqG = utils.mgrid(acq_rd)
acqX = utils.mesh2points(acqG,acq_rd[:,2])

mn_rd = np.array([[-1,1,15],[-1,1,15],[-1,1,15]])
mnG   = utils.mgrid(mn_rd)
meanX  = utils.mesh2points(mnG,mn_rd[:,2])


def invalid_region(X):
    return np.array([False]*len(X))
