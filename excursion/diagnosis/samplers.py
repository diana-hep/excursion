import numpy as np
import itertools

from .. import  utils

def regular_grid_generator(scandetails, central_range = [5,20], nsamples_per_grid = 15, min_points_per_dim = 2):
    ndim = len(scandetails.plot_rangedef[:,2])
    grids = set(x for y in range(*central_range) for x in itertools.combinations_with_replacement([y,y-1,y-2],ndim))
    grids = [g for g in grids if np.all(np.array(g) >= min_points_per_dim)]
    grids = sorted(grids, key=lambda k: np.product(k))
    def makegrids(sizes):
        for s in range(nsamples_per_grid):
            los = np.random.uniform(0,0.1, size = ndim)
            his = np.random.uniform(0.9, 1.0, size =ndim)
            yield np.array(np.meshgrid(*[np.linspace(los[i],his[i],sizes[i]) for i in range(ndim)]))
    for g in grids:
        for mesh in makegrids(g):
            X = utils.mesh2points(mesh, mesh.shape[1:])
            for i in range(ndim):
                vmin, vmax = scandetails.plot_rangedef[i][0], scandetails.plot_rangedef[i][1]
                X[:,i] = X[:,i]*(vmax-vmin) + vmin
            X = X[~scandetails.invalid_region(X)]
            yield X,g

def latin_hypercube_generator(scandetails, nsamples_per_npoints = 50, point_range = [4, 100]):
    ndim = len(scandetails.plot_rangedef[:,2])
    import pyDOE
    for npoints in range(*point_range):
        for s in range(nsamples_per_npoints):
            sample_n = npoints
            while True:
                print('sampling',sample_n)
                X = pyDOE.lhs(ndim, samples=sample_n)

                for i in range(ndim):
                    vmin, vmax = scandetails.plot_rangedef[i][0], scandetails.plot_rangedef[i][1]
                    X[:,i] = X[:,i]*(vmax-vmin) + vmin
                len_before = len(X)
                X = X[~scandetails.invalid_region(X)]
                len_after = len(X)
                if not len_after >= npoints: #invalid might throw out points, so sample until we get what we want
                    sample_n = int(sample_n * float(len_before)/float(len_after))
                    print('increasing to %s' % sample_n)
                    continue
                break
            yield X[:npoints],None
