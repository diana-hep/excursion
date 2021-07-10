#!/usr/bin/python

from . import getMuLimitBatch
import torch
import numpy as np
from excursion.utils import mgrid, mesh2points


x = np.array( [ [1200.0, 850.0], [1300.0, 950.0] ] )

limits = getMuLimitBatch.getMuLimitBatch(x, "V", "exp", "testRun")

print("limits are: {}".format(limits))


####
true_functions = [limits]

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
import numpy as np

import sys
sys.path.append("/afs/ipp-garching.mpg.de/common/soft/anaconda/amd64_generic/3/5.3.0/lib/python3.7/site-packages")
import matplotlib

import ROOT

import getMuLimitBatch

x = np.array( [ [1200.0, 850.0], [1300.0, 950.0] ] )

limits = getMuLimitBatch.getMuLimitBatch(x, "V", "exp", "testRun")

print("limits are: {}".format(limits))
'''
