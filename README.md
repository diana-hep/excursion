# `excursion` — Efficient Excursion Set Estimation 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1634427.svg)](https://zenodo.org/badge/latestdoi/146087019)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/diana-hep/excursion/master?filepath=examples%2FBinder.ipynb)
[![Build Status](https://travis-ci.com/diana-hep/excursion.svg?branch=master)](https://travis-ci.com/diana-hep/excursion)

This package implements a Bayesian Optimization procedure based on Gaussian Processes to efficiently determine excursion sets (or equivalently iso-surfaces) of one or many expensive black-box functions.

## Installation and Example

Install via `pip install excursion==0.0.1a0`.

To estimate excursion sets for `N_FUNCS=2` functions simultaneously run:

```python
from excursion import ExcursionProblem
import excursion.optimize as optimize
import numpy as np
import scipy.stats

N_UPDATES = 10
N_BATCH = 2

def expensive_func(X):
    return np.atleast_1d(scipy.stats.multivariate_normal.pdf(X,mean = [0.5,0.5], cov = np.diag([0.2,0.3])))

scandetails = ExcursionProblem([expensive_func], ndim = 2)
X,y_list,gps = optimize.init(scandetails)
for index in range(N_UPDATES):
    print('next')
    newX = optimize.suggest(gps, X, scandetails, batchsize=N_BATCH)
    print(newX)
    X,y_list,gps  = optimize.evaluate_and_refine(X,y_list,newX,scandetails)
```

## Ex: Finding two-dimensional Contours in High-Energy Physics

In this example, modeled after typical exclusion contours of high-energy physics searches, we are insterested in estimating two excursion sets

1. the **excluded** set of points -- theories of physics incompatible with the data
2. the **non-excluded** set of points -- theories that are still viable.

Typically two simultaneous labels can be assigned, the *expected* and *observed* status of a given theory. The label can be computed through computationally expensive Monte Carlo simulation. Points to be simulated are therefore picked to most efficiently estimate both the *expected* and *observed* excursion sets.

<img src="./assets/truth.png" width=200/>

### Point Seqeuence

<img src="./assets/example.gif" width=600/>

## Talks:

* 4th Reinterpretation Workshop [Slides](https://indico.cern.ch/event/702612/contributions/2958660/attachments/1649620/2638023/Contours.pdf)

## Authors

* Lukas Heinrich, CERN
* Gilles Louppe, U Liege
* Kyle Cranmer, NYU
