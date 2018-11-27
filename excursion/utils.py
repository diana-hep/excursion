from scipy.stats import norm
import numpy as np

def point_entropy(mu_stds, thresholds):
    thresholds = np.concatenate([[-np.inf],thresholds,[np.inf]])

    entropies = []
    for mu,std in mu_stds:
        entropy = 0
        for j in range(len(thresholds) - 1):
            p_within = norm(mu, std).cdf(thresholds[j+1]) - norm(mu, std).cdf(thresholds[j])
            p_within[p_within < 1e-9] = 1e-9
            p_within[p_within > 1-1e-9] = 1-1e-9
            entropy -= p_within * np.log(p_within)
        entropies.append(entropy)
    return np.mean(np.stack(entropies), axis=0)

def mesh2points(grid,npoints_tuple):
    ndim = len(npoints_tuple)
    X = np.moveaxis(grid,0,ndim).reshape(int(np.product(npoints_tuple)),ndim)
    return X

def points2mesh(X,npoints_tuple):
    ndim = len(npoints_tuple)
    grid = np.moveaxis(X.reshape(*(npoints_tuple +[ndim,])),ndim,0)
    return grid

def mgrid(rangedef):
    _rangedef = np.array(rangedef, dtype='complex128')
    slices = [slice(*_r) for _r in _rangedef]
    return np.mgrid[slices]

def values2mesh(values, rangedef, invalid, invalid_value = np.nan):
    grid = mgrid(rangedef)
    allX = mesh2points(grid,rangedef[:,2])
    allv = np.zeros(len(allX))
    inv  = invalid(allX)
    allv[~inv]  = values
    if np.any(inv):
        allv[inv]  = invalid_value
    return allv.reshape(*map(int,rangedef[:,2]))
