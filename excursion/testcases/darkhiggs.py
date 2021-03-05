import json
import base64
import pickle
import pkg_resources
import numpy as np
from .. import utils
from .. import ExcursionProblem

datafile = pkg_resources.resource_filename('excursion','testcases/data/darkhiggsdata.json')

scandata = json.load(open(datafile))

zfilter = [.05, 0.5]
truthX = []
truthy = []
for i,(k,v) in enumerate(scandata.items()):
    pointdata = json.loads(base64.b64decode(k).decode('ascii'))
    x,y,z = pointdata['mzp'], pointdata['mdm'], pointdata['gq']
    try:
        r = np.log(v[0]['result']['xsec'])
    except KeyError:
        continue
        r = 2.0
    if not (zfilter[0] < z < zfilter[1]): continue
    truthX.append([x,y,z])
    truthy.append(r)

truthX = np.array(truthX)
truthy = np.array(truthy)

import sklearn.preprocessing
scaler = sklearn.preprocessing.MinMaxScaler()
scaler.fit(truthX)
truthX = scaler.transform(truthX)


def truth(denseX):
    from scipy.interpolate import griddata
    return griddata(truthX,truthy,denseX)

def invalid_region(X):
    return np.isnan(truth(X))

# gpfile = pkg_resources.resource_filename('excursion','testcases/data/darkhiggsgp.pickle')

# gp = pickle.load(open(gpfile,'rb'))

# truth = lambda X: gp.predict(X)

npoints = [50,50,50]
iso_xsec = ExcursionProblem(
    [truth], [-9.5], ndim = 3,
    plot_npoints=[50,50,50], invalid_region = invalid_region
)
