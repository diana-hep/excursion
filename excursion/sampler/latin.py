import pyDOE
# # # To be implemented


def latin_sample_n(rangedef, invalid_region, npoints, ndim):
    sample_n = npoints
    while True:
        X = pyDOE.lhs(ndim, samples=sample_n)

        for i in range(ndim):
            vmin, vmax = rangedef[i][0], rangedef[i][1]
            X[:,i] = X[:,i]*(vmax-vmin) + vmin
        len_before = len(X)
        X = X[~invalid_region(X)]
        len_after = len(X)
        if not len_after >= npoints: #invalid might throw out points, so sample until we get what we want
            factor = float(len_before)/float(len_after) if len_after else 2
            sample_n = int(sample_n * factor)
            continue
        return X[:npoints]

#
# def latin_hypercube_generator(scandetails, nsamples_per_npoints = 50, point_range = [4, 100]):
#     ndim = len(scandetails.rangedef[:, 2])
#     for npoints in range(*point_range):
#         for s in range(nsamples_per_npoints):
#             yield latin_sample_n(scandetails,npoints,ndim), None
