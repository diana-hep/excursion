# `excursion` — Efficient Excursion Set Estimation 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1634427.svg)](https://zenodo.org/badge/latestdoi/146087019)

This package implements a Bayesian Optimization procedure based on Gaussian Processes to efficiently determine excursion sets (or equivalently iso-surfaces) of one or many expensive black-box functions.

## Installation and Example

Install via `pip install excursion==0.0.1a0` and run: 

```python
for index in range(N_UPDATES):
    gps = [excursion.get_gp(X,y_list[i]) for i in range(N_FUNCS)]
    newx, acqvals = excursion.optimize.gridsearch(gps, X, scandetails)
    newys_list = [expensive_functions[i](np.asarray([newx])) for i in range(N_FUNCS)]
    for i,newys in enumerate(newys_list):
        y_list[i] = np.concatenate([y_list[i],newys])
    X = np.concatenate([X,np.array([newx])])
```

## Ex: Finding two-dimensional Contours in High-Energy Physics

In this example, modeled after typical exclusion contours of high-energy physics searches, we are insterested in estimating two excursion sets

1. the **excluded** set of points -- theories of physics incomptaible with the data
2. the **non-excluded** set of points -- theories that are still viable.

Typically two simultaneous labels can be assigned, the *expected* and *observed* status of a given theory. The label can be computed through computationally expensive Monte Carlo simulation. Points to be simulated are therefor picked to most efficiently estimate both the *expected* and *observed* excursion sets.

<img src="./assets/truth.png" width=200/>

### Point Seqeuence

<img src="./assets/example.gif" width=600/>

## Authors

* Lukas Heinrich, CERN
* Gilles Louppe, U Liege
* Kyle Cranmer, NYU