import json
import base64
import pkg_resources
import numpy as np
from .. import utils

datafile = pkg_resources.resource_filename('excursion','testcases/data/darkhiggsdata.json')

scandata = json.load(open(datafile))

zfilter = [.05, 1.0]
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

def invalid_region(x):
    return np.array([False]*len(x))

def truth(denseX):
    z_val = 0.5
    from scipy.interpolate import griddata
    w = np.concatenate([denseX,z_val*np.ones(len(denseX)).reshape(-1,1)],axis=-1)
    return griddata(truthX,truthy,w)


truth_functions = [truth]

plot_rangedef = np.array([[0.1,0.9,101],[0.1,0.9,101]])
plotG = utils.mgrid(plot_rangedef)
plotX = utils.mesh2points(plotG,plot_rangedef[:,2])

thresholds = [-14.5,-13.5, -12.5]

acq_rd = np.array([[0.1,0.90,41],[0.1,0.90,41]])
acqG = utils.mgrid(acq_rd)
acqX = utils.mesh2points(acqG,acq_rd[:,2])

mn_rd = np.array([[0.1,0.90,41],[0.1,0.90,41]])
mnG   = utils.mgrid(mn_rd)
meanX  = utils.mesh2points(mnG,mn_rd[:,2])
