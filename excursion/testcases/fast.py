import numpy as np
from .. import utils
from .. import ExcursionProblem

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


#shift to simulate mismatch between exp / obs
shifted_truth = lambda X: truth(X-0.05)

bounding_box = [[0,1.5],[0,1.5]]
single_function = ExcursionProblem([truth],[0.0],ndim = 2, bounding_box = bounding_box)
two_functions = ExcursionProblem([truth,shifted_truth],[0.0],ndim = 2, bounding_box = bounding_box)
