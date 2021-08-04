import json
import base64
import pkg_resources
import numpy as np
import torch
import sklearn.preprocessing
from excursion.utils import mgrid, mesh2points

datafile = pkg_resources.resource_filename('excursion','unfactored/data/darkhiggsdata.json')

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


scaler = sklearn.preprocessing.MinMaxScaler()
scaler.fit(truthX)
truthX = scaler.transform(truthX)

def invalid_region(x):
    return np.array([False]*len(x))

def truth(denseX):
    from scipy.interpolate import griddata
    grid_data = torch.from_numpy(griddata(truthX,truthy,denseX))
    return grid_data


thresholds = torch.Tensor([-9.5])
true_functions = [truth]
n_dims = 3

rangedef = np.array([[0.1,0.9,41],[0.1,0.9,41],[0.1,0.9,41]])
plot_meshgrid = mgrid(rangedef)
X_plot = mesh2points(plot_meshgrid, rangedef[:, 2])
X = torch.from_numpy(X_plot)


