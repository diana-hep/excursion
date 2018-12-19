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

def invalid_region(x):
    return np.array([False]*len(x))

plot_rangedef = np.array([[0.0,1.5,101],[0.0,1.5,101]])
plotG = utils.mgrid(plot_rangedef)
plotX = utils.mesh2points(plotG,plot_rangedef[:,2])

functions = [truth]
thresholds = [0.0]


def acqX():
    print('requested acqX')
    return np.random.uniform(plot_rangedef[:,0],plot_rangedef[:,1], size = (500,2))

def meanX():
    print('requested meanX')
    return np.random.uniform(plot_rangedef[:,0],plot_rangedef[:,1], size = (500,2))

def test_data():
    X = plotX
    y_list = [func(X) for func in functions]
    return X,y_list