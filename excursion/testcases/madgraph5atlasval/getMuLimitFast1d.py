#!/usr/bin/python

import pickle
import numpy as np
import torch
from excursion.utils import mgrid, mesh2points
import os

def shapeMuLimit(muLimit):
    muLimit = np.log( muLimit + 1e-9 )
    return muLimit


def getMuLimitFast1d(xx):
    inFile      = open("../excursion/testcases/madgraph5atlasval/dileptonLimits_200221_av_mdm500.pickle", "rb")
    content     = pickle.load( inFile )
    xGrid       = content["X"]
    muLimitGrid = content["muLimits"]
    muLimits    = []
    for x in xx:
        muLimit = -1.0
        dx      = 1e2
        for iXg in range(len(xGrid)):
            thisDx = abs( x[0] - xGrid[iXg][0] )
            if thisDx < dx:
                dx = thisDx
                muLimit = muLimitGrid[iXg]
        muLimits.append( shapeMuLimit(muLimit) )
    return torch.Tensor(muLimits)

true_functions = [getMuLimitFast1d]

thresholds = torch.Tensor([0.0])

n_dims = 1

rangedef = np.array([[0.0, 1.0, 100]])

plot_meshgrid = mgrid(rangedef)

X_plot = mesh2points(plot_meshgrid, rangedef[:, 2])
plot_X = torch.from_numpy(mesh2points(plot_meshgrid, rangedef[:, 2]))

X = torch.from_numpy(X_plot)

def invalid_region(x):
    return np.array([False] * len(x))

'''
import pickle
import numpy as np

def shapeMuLimit(muLimit):
    muLimit = np.log( muLimit + 1e-9 )
    return muLimit

#--------------------------------------------------

def getMuLimitFast1d(xx):
    inFile      = open("dileptonLimits_200221_av_mdm500.pickle", "rb")
    content     = pickle.load( inFile )
    xGrid       = content["X"]
    muLimitGrid = content["muLimits"]
    muLimits    = []
    for x in xx:
        muLimit = -1.0
        dx      = 1e2
        for iXg in range(len(xGrid)):
            thisDx = abs( x[0] - xGrid[iXg][0] )
            if thisDx < dx:
                dx = thisDx
                muLimit = muLimitGrid[iXg]
        muLimits.append( shapeMuLimit(muLimit) )
    return muLimits
'''
